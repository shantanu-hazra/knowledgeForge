import asyncio
import json

from llm.client import LLM
from llm.schemas import Message
from database.session import Session
from agent.memory import load, summarize_if_needed, save
from agent.prompt_builder import build_prompt
from agent.agent_executor import run_plan
from agent.planner import PlannerDecision, Planner
from agent.agents import writer_agent

MAX_PLANNING_ROUNDS = 5  # hard stop against infinite agent-call loops

# NOTE: deliberately NOT using prompt_builder.DEFAULT_SYSTEM_PROMPT here.
# That prompt instructs the model to call calculator/weather/
# knowledge_base_search tools, but nothing in this file ever passes
# `tools=` to LLM.chat() (planner.decide() builds its own tool-free
# planning prompt; _direct_answer() below also calls chat() with no
# tools). Feeding that prompt into either path makes the model attempt
# a tool call the request structurally can't support, which Groq
# rejects with "Tool choice is none, but model called a tool". If
# those three tools get wired into a real `tools=` call somewhere,
# reintroduce DEFAULT_SYSTEM_PROMPT there specifically — not here.
BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer directly and concisely, "
    "grounded in the conversation so far."
)


class Agent:
    def __init__(self):
        self.llm = LLM()
        self.planner = Planner(llm=self.llm)   # shares the same LLM client

    async def run(self, req) -> str:
        print(f"[agent] user query: {req.message}")

        history = load(
            session_id=getattr(req, "conversation_id", None),
            user_id=req.user_id,
        )
        session: Session = history.session
        session.add_user_message(req.message)

        prompt = build_prompt(session, user_message=req.message, system_prompt=BASE_SYSTEM_PROMPT)

        # ---- planner decides which agent(s) are needed, and their tasks ----
        decision = await self._decide(prompt)

        results: dict = {}
        rounds = 0
        while decision.tasks_required and rounds < MAX_PLANNING_ROUNDS:
            round_results = await run_plan(decision.tasks)
            results.update(round_results)

            # Let the planner see what came back so it can decide whether
            # more tasks are needed (e.g. reviewer_agent rejected the
            # draft and writer_agent needs another pass). There's no
            # OpenAI tool-call/tool-result message to replay anymore —
            # just tell the planner what happened, in plain content.
            prompt.append(
                Message(
                    role="user",
                    content=(
                        "Results from the agent tasks just run:\n"
                        f"{json.dumps(round_results, default=str)}\n\n"
                        "Given these results, are further agent tasks needed "
                        "to fully answer the original question? If not, set "
                        "tasks_required to false."
                    ),
                )
            )
            decision = await self._decide(prompt)
            rounds += 1

        final_text = self._extract_final_answer(results)
        if final_text is None and results:
            # Evidence was gathered this turn (retrieval_agent /
            # research_agent / analysis_agent ran) but the planner never
            # routed to writer_agent to turn it into an answer — e.g. it
            # decided tasks_required=False right after seeing raw
            # retrieval chunks, treating "we have chunks" as "we're
            # done" instead of "now synthesize them". Falling through to
            # _direct_answer() in that case is the actual bug this
            # branch fixes: _direct_answer() only sees `results` as a
            # raw JSON blob buried in conversation history, under
            # BASE_SYSTEM_PROMPT's generic "grounded in the conversation
            # so far" instruction — nowhere near as strict as
            # writer_agent's own system prompt, which explicitly
            # forbids inventing facts and requires staying grounded in
            # the evidence block. Rather than let gathered evidence go
            # unused (or under-grounded) whenever the planner stops one
            # step early, force a real writer_agent synthesis pass over
            # whatever was gathered.
            writer_output = await asyncio.to_thread(
                writer_agent.run, {"query": req.message}, results
            )
            final_text = writer_output["answer"]
        elif final_text is None:
            # No agent ran at all this turn — nothing to ground on, so a
            # plain conversational answer is the correct fallback here.
            final_text = await self._direct_answer(prompt)

        session.add_assistant_message(final_text)
        history = summarize_if_needed(history)
        save(history)

        return final_text, session.conversation_id

    async def _decide(self, prompt: list[Message]) -> PlannerDecision:
        """Offloads planner.decide() (which itself calls llm.chat) to a thread."""
        decision = await asyncio.to_thread(self.planner.decide, prompt)
        print(
            f"[agent] planner decision: tasks_required={decision.tasks_required} "
            f"agents={[t.agent_name for t in (decision.tasks or [])]} "
            f"depends_on={[t.depends_on for t in (decision.tasks or [])]} "
            f"reasoning={decision.reasoning!r}"
        )
        return decision

    @staticmethod
    def _extract_final_answer(results: dict) -> str | None:
        """
        Pulls the user-facing answer out of writer_agent's output, if it
        ran this turn. Mirrors the same lookup reviewer_agent.py already
        does over `results` — {"answer": ...} is writer_agent's shape.
        If writer_agent ran more than once (e.g. after a reviewer
        rejection), the later dict entry naturally wins since dict
        iteration follows insertion order and `results` is updated per
        round in run().
        """
        answer = None
        for output in results.values():
            if isinstance(output, dict) and "answer" in output:
                answer = output["answer"]
        return answer

    async def _direct_answer(self, prompt: list[Message]) -> str:
        """
        Plain (non-planning) LLM call for turns where no agent tasks were
        ever required — the planner's own LLM call only ever produces
        plan-JSON now, so it can't double as the final answer the way a
        single OpenAI tool-calling response used to.
        """
        response = await asyncio.to_thread(self.llm.chat, prompt)
        return response.content