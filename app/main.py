from fastapi import FastAPI
from app.routers import webhook

app = FastAPI(title="LINE Repair Bot")

app.include_router(webhook.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
