from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat.router import router as chat_router
from api.document.router import router as rag_router
from database.session import DB_PATH

from database.crud import init_db
app = FastAPI()

# Allows the static test console (opened via file:// or a separate
# dev server) to call this API from the browser. Browsers preflight
# any cross-origin POST with a JSON body via OPTIONS first — with no
# CORS middleware registered, FastAPI has no handler for that OPTIONS
# request at all, which is why it came back 405 rather than reaching
# the real POST /chat handler.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend origin(s) before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db(DB_PATH)

app.include_router(chat_router)
app.include_router(rag_router)


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=False)