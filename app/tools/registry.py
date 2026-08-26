"""
Auto-generates OpenAI/Groq-style tool schemas from pydantic models,
instead of hand-writing each JSON schema dict.

Each tool = (name, description, pydantic args model, callable).
The registry builds both TOOLS (name -> callable) and TOOL_SCHEMAS
(the list passed to `tools=` in the chat completion call) from the
same source of truth, so they can't drift apart.
"""

import json
from typing import Callable
from pydantic import BaseModel, ValidationError

from tools.weather import get_weather
from tools.web_search import web_search
from tools.calculator import add, sub, mult, divide, pct
from tools.knowledge_base import knowledge_base_search


# ---- 1. Define input schemas for each tool -------------------------------

class WeatherArgs(BaseModel):
    latitude: float
    longitude: float
    hourly: str = "temperature_2m"


class SearchArgs(BaseModel):
    query: str
    max_results: int = 5


class TwoNumberArgs(BaseModel):
    a: float
    b: float

class KnowledgeBaseArgs(BaseModel):
    query: str

# ---- 2. Register each tool: name -> (function, args_model, description) --

class Tool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable

from enum import Enum

class Operation(str, Enum):
    add = "add"
    sub = "sub"
    mult = "mult"
    divide = "divide"
    pct = "pct"

class CalculatorArgs(BaseModel):
    operation: Operation
    a: float
    b: float

_OPS = {
    Operation.add: add,
    Operation.sub: sub,
    Operation.mult: mult,
    Operation.divide: divide,
    Operation.pct: pct,
}

def calculator(operation: Operation, a: float, b: float):
    return _OPS[operation](a, b)

REGISTRY: list[Tool] = [
    Tool(
        name="weather",
        description="Get hourly forecast data for a location.",
        args_model=WeatherArgs,
        fn=get_weather,
    ),
    Tool(
        name="search",
        description="Use this to search the web for current information, facts, or anything the user wants looked up online. Use for any query starting with 'what is', 'who is', 'search for', etc.",
        args_model=SearchArgs,
        fn=web_search,
    ),
    Tool(
        name="calculator",
        description="Perform a basic arithmetic operation (add, sub, mult, divide, pct) on two numbers.",
        args_model=CalculatorArgs,
        fn=calculator,
    ),
    Tool(
        name="knowledge_base_search",
        description=(
            "Search the internal knowledge base, which contains user-specific "
            "and organizational documents (e.g. offer letters, policies, HR "
            "records, contracts). Use this tool for ANY question that might be "
            "answered by a document in the knowledge base — including questions "
            "about the user's own personal details such as salary, CTC, "
            "designation, or employment terms. Do not assume you lack access; "
            "always attempt this search before saying information is unavailable."
        ),
        args_model=KnowledgeBaseArgs,
        fn=knowledge_base_search,
    ),
]

TOOLS: dict[str, Tool] = {t.name: t for t in REGISTRY}


# ---- 3. Build the schema list the LLM API expects -------------------------

def build_tool_schemas() -> list[dict]:
    schemas = []
    for tool in REGISTRY:
        schema = tool.args_model.model_json_schema()
        # pydantic emits "title" fields the API doesn't need; strip for cleanliness
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)

        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        })
    return schemas


TOOL_SCHEMAS = build_tool_schemas()


# ---- 4. Dispatcher: validate args against the model, then call the fn -----

class ToolDispatcher:
    def __init__(self, registry: dict[str, Tool] = TOOLS):
        self.registry = registry

    def call(self, name: str, arguments: dict) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            validated = tool.args_model(**arguments)
        except ValidationError as e:
            return json.dumps({"error": f"Invalid arguments for '{name}': {e}"})

        try:
            result = tool.fn(**validated.model_dump())
        except Exception as e:
            import traceback
            print(f"[ToolDispatcher] '{name}' failed:")
            traceback.print_exc()
            return json.dumps({"error": f"Tool '{name}' failed: {e}"})

        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return json.dumps({"result": result})


if __name__ == "__main__":
    import pprint
    pprint.pprint(TOOL_SCHEMAS)

    dispatcher = ToolDispatcher()
    print(dispatcher.call("add", {"a": 2, "b": 3}))
    print(dispatcher.call("weather", {"latitude": 52.52, "longitude": 13.41}))
    print(dispatcher.call("frobnicate", {}))