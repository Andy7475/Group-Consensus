"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from group_consensus.api.routes import deliberation, sessions, voting

app = FastAPI(
    title="Group Consensus API",
    description=(
        "AI-mediated group deliberation combining the Habermas Machine approach "
        "with Polis-style opinion clustering."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(deliberation.router)
app.include_router(voting.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
