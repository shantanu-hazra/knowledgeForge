from fastapi import APIRouter
from agent.supervisor import Agent
from api.models import ChatRequest, ChatResponse
from database.crud import init_db, DB_PATH

router = APIRouter()

@router.post("/chat",response_model=ChatResponse)
async def chat(req: ChatRequest):
    init_db(DB_PATH)  # Initialize the database connection
    agent=Agent()
    reply, conversation_id = await agent.run(req)
    print(f"Reply: {reply}")
    return ChatResponse(reply=reply,conversation_id=conversation_id)