"""Task 9: the notes and tasks MCP server. Run by docker compose / CI as:  python src/app/mcp_server/server.py --http  (port 9000, path /mcp)

The storage is PROVIDED (taskstore.py: TaskStore), the user lookup is PROVIDED (app/users.py: UserStore.user_for_token).
You write the integration: register the eight tools below on `mcp`, and make every call run AS THE CALLING USER.

How the server knows the user (task 9)
  The agent forwards the logged-in user's token on every MCP call as the HTTP header  Authorization: Bearer <token>  (you write that
  half in agent.py, task 9). A tool declares a `ctx: Context` parameter; FastMCP injects it and the HTTP request is
  ctx.request_context.request (Starlette: .headers). Write ONE helper `current_user_id(ctx) -> str` that
    - reads the bearer token, looks it up with UserStore().user_for_token(token)
    - raises ToolError("ERROR[UNAUTHENTICATED]: <what>. Next step: log in and retry.") when the header is missing, malformed or unknown
  and call it first in every tool. Never take a user id from a tool argument: the model must not be able to name another user.

The eight tools (names and arguments are fixed; each returns json.dumps(<what the TaskStore method returned>), a string)
  add_note(title: str, body: str)               list_notes()                          delete_note(note_id: int)
  add_task(title: str, details: str = "")       list_tasks(status: str = "")          set_task_status(task_id: int, status: str)
  delete_task(task_id: int)                     task_summary()
  status is "todo" | "in_progress" | "done". TaskStore raises TaskError for bad input: re-raise it as ToolError(str(error)).
  The docstring of each tool is its prompt for the model: say WHEN to use it and describe the arguments.
"""
import json  # noqa: F401  (you will need it)
import os
import sys

from mcp.server.fastmcp import Context, FastMCP  # noqa: F401
from mcp.server.fastmcp.exceptions import ToolError  # noqa: F401

from app.mcp_server.taskstore import TaskError, TaskStore  # noqa: F401
from app.users import UserStore  # noqa: F401

mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()


def current_user_id(ctx: Context) -> str:
    authorization = ctx.request_context.request.headers.get("authorization", "")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ToolError("ERROR[UNAUTHENTICATED]: A valid bearer token is required. Next step: log in and retry.")
    user = UserStore().user_for_token(parts[1])
    if user is None:
        raise ToolError("ERROR[UNAUTHENTICATED]: A valid bearer token is required. Next step: log in and retry.")
    return user["user_id"]


def _json_task_result(method, *args) -> str:
    try:
        return json.dumps(method(*args))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
async def add_note(title: str, body: str, ctx: Context) -> str:
    """Save a note for the calling user. Use for information they want to remember without a task status.

    Args:
        title: Short name for the note.
        body: The note's content.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.add_note, user_id, title, body)


@mcp.tool()
async def list_notes(ctx: Context) -> str:
    """List the calling user's saved notes. Use when asked what notes they have stored."""
    user_id = current_user_id(ctx)
    return _json_task_result(store.list_notes, user_id)


@mcp.tool()
async def delete_note(note_id: int, ctx: Context) -> str:
    """Delete one of the calling user's notes.

    Args:
        note_id: The note id returned by list_notes.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.delete_note, user_id, note_id)


@mcp.tool()
async def add_task(ctx: Context, title: str, details: str = "") -> str:
    """Create a task for the calling user with initial status todo.

    Args:
        title: Short task description.
        details: Optional additional task details.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.add_task, user_id, title, details)


@mcp.tool()
async def list_tasks(ctx: Context, status: str = "") -> str:
    """List the calling user's tasks, optionally filtered by status (todo, in_progress, or done).

    Args:
        status: Status to filter by, or empty to list every task.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.list_tasks, user_id, status)


@mcp.tool()
async def set_task_status(task_id: int, status: str, ctx: Context) -> str:
    """Move one of the calling user's tasks one allowed step (todo, in_progress, or done).

    Args:
        task_id: The task id returned by list_tasks.
        status: The new status: todo, in_progress, or done.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.set_task_status, user_id, task_id, status)


@mcp.tool()
async def delete_task(task_id: int, ctx: Context) -> str:
    """Delete one of the calling user's tasks.

    Args:
        task_id: The task id returned by list_tasks.
    """
    user_id = current_user_id(ctx)
    return _json_task_result(store.delete_task, user_id, task_id)


@mcp.tool()
async def task_summary(ctx: Context) -> str:
    """Summarize the calling user's task counts by status. Use when asked for task progress."""
    user_id = current_user_id(ctx)
    return _json_task_result(store.summary, user_id)


if __name__ == "__main__":
    # stdio cannot carry the Authorization header: the agent talks to this server over HTTP (MCP_URL)
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
