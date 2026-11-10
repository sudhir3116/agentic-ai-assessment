"""SQLite-backed LangGraph checkpoints, held open for the API lifespan."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        yield saver
