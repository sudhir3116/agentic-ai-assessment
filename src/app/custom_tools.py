"""Tasks 7 and 8: four tools. Names, arguments, outputs, errors, metric and log names are FIXED (the autograder checks them).

The heavy parts are PROVIDED (kb/store.py: stats / outline / delete_document, kb/pipeline.py: ingest_file).
You write the tool wrappers: signature + docstring (the docstring is the prompt: say WHEN to use it, describe each argument),
validation, ToolException with `ERROR[CODE]: what happened. Next step: what to do.`, the metric, the log line, `handle_tool_error = True`.
All four are `async def` (blocking calls go through asyncio.to_thread). They act on the CALLING USER's knowledge base:
`runtime.context.user_id` (task 6), never on another user's. build_custom_tools(kb) returns all four.
Look at kb/tools.py for the pattern (ToolRuntime, ToolException, handle_tool_error).

---------------------------------------------------------------------------------- task 7.1  kb_stats(top_keywords: int = 5) -> str
  output  "<D> documents, <S> sections, <C> chunks. Top keywords: k1, k2, ..."   from kb.stats(user_id); at most `top_keywords` keywords
          empty knowledge base: "The knowledge base is empty."
  error   top_keywords outside 1..20  ->  ERROR[INVALID_INPUT]

---------------------------------------------------------------------------------- task 7.2  document_outline(doc_name: str) -> str
  The structure of one document: use it when the user asks what a document contains, how it is organised, or which section covers
  something (then search that section). Resolve doc_name with kb.resolve_doc_name(user_id, doc_name), find the doc_id in
  kb.documents(user_id), read the sections with kb.outline(user_id, doc_id).
  output  first line   "Outline of '<exact file name>' (<N> sections):"          (always the word "sections")
          then one line per section, in reading order, 1-based position:
              "<i>. [L<level>] <heading> (p.<a>-<b>): <summary>"
          the page part is " (p.<a>-<b>)" when page_start != page_end, " (p.<a>)" when they are equal, and left out when the
          document has no pages (page_start is None); then ": <summary>"
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 7.3  observability, every call, ok AND error
  metric  kb_stats: counter "kbstats.calls"    document_outline: counter "outline.calls"
          record with attributes {"status": "ok" | "error"}  (Prometheus: agent_outline_calls_total, ...)
          metrics.get_meter("agent").create_counter(...)   (see how app/telemetry.py creates TOOL_CALLS)
  log     log.info("kb_stats_done", tool="kb_stats", status=..., duration_ms=<int>)
          log.info("document_outline_done", tool="document_outline", status=..., duration_ms=<int>, sections=<int, 0 on error>)
          (`from app.telemetry import log`; never log document text)
  Grafana (recommended, not graded): src/app/dashboards/outline.json with a panel for outline.calls by status and a logs panel
          ({service_name="extraction-agent"} | json | event="document_outline_done").

---------------------------------------------------------------------------------- task 8.1  delete_document(doc_name: str) -> str
  Removes one document from the user's knowledge base. doc_name is a file name from the KNOWLEDGE BASE list.
  Resolve it with kb.resolve_doc_name(user_id, doc_name); find the document's doc_id in kb.documents(user_id);
  delete with kb.delete_document(user_id, doc_id).
  output  "Deleted '<exact file name>'. <N> documents left."
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 8.2  update_document(doc_name: str, source: str) -> str
  Replaces a document with a new version the user uploaded. source = the file_id of the NEW upload (like ingest_document).
  Order matters: ingest the new file FIRST under the OLD file name (kb/pipeline.ingest_file, files.resolve), and only when that
  succeeded delete the old version by its doc_id: a failed update must leave the old document untouched.
  If the new file is identical to a document already in the knowledge base ingest_file returns it as `existing`: change nothing.
  output  "Updated '<exact file name>': <S> sections, <C> chunks."      identical content: "'<name>' is already up to date."
  errors  unknown doc_name -> ERROR[UNKNOWN_DOCUMENT];  bad/unknown file_id -> use app.errors.to_tool_exception on the DocError
          (it yields ERROR[FILE_NOT_FOUND]).
"""
import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log

_meter = metrics.get_meter("agent")
_KBSTATS_CALLS = _meter.create_counter("kbstats.calls", unit="1")
_OUTLINE_CALLS = _meter.create_counter("outline.calls", unit="1")


def _unknown_document(doc_name: str, documents: list[dict]) -> ToolException:
        names = ", ".join(f'"{document["filename"]}"' for document in documents) or "(none)"
        return ToolException(
                f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. Documents in your knowledge base: {names}. "
                "Next step: retry with one of these names."
        )


def build_custom_tools(kb: KnowledgeBase) -> list:
        @tool
        async def kb_stats(runtime: ToolRuntime[Ctx], top_keywords: int = 5) -> str:
                """Summarize the calling user's knowledge base. Use when asked how many documents or sections it contains, or which topics recur.

                Args:
                        top_keywords: How many of the most common document keywords to include (1-20).
                """
                started = time.perf_counter()
                status = "error"
                try:
                        if not 1 <= top_keywords <= 20:
                                raise ToolException("ERROR[INVALID_INPUT]: top_keywords must be between 1 and 20. Next step: choose a value from 1 to 20.")
                        stats = await asyncio.to_thread(kb.stats, runtime.context.user_id)
                        if not stats["documents"]:
                                result = "The knowledge base is empty."
                        else:
                                keywords = ", ".join(stats["keywords"][:top_keywords])
                                result = (f"{stats['documents']} documents, {stats['sections']} sections, {stats['chunks']} chunks. "
                                                                                                          f"Top keywords: {keywords}.")
                        status = "ok"
                        return result
                except Exception as exc:
                        if isinstance(exc, ToolException):
                                raise
                        raise to_tool_exception(exc) from exc
                finally:
                        _KBSTATS_CALLS.add(1, {"status": status})
                        log.info("kb_stats_done", tool="kb_stats", status=status,
                                         duration_ms=int((time.perf_counter() - started) * 1000))

        @tool
        async def document_outline(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
                """Show a document's section structure. Use when asked what a document contains, how it is organised, or which section covers a topic.

                Args:
                        doc_name: The document's file name from the knowledge-base list.
                """
                started = time.perf_counter()
                status = "error"
                section_count = 0
                try:
                        user_id = runtime.context.user_id
                        documents = await asyncio.to_thread(kb.documents, user_id)
                        exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
                        if exact is None:
                                raise _unknown_document(doc_name, documents)
                        document = next(item for item in documents if item["filename"] == exact)
                        sections = await asyncio.to_thread(kb.outline, user_id, document["doc_id"])
                        section_count = len(sections)
                        lines = [f"Outline of '{exact}' ({section_count} sections):"]
                        for position, section in enumerate(sections, start=1):
                                page_start, page_end = section["page_start"], section["page_end"]
                                page_text = ""
                                if page_start is not None:
                                        page_text = f" (p.{page_start})" if page_start == page_end else f" (p.{page_start}-{page_end})"
                                lines.append(
                                        f"{position}. [L{section['level']}] {section['heading']}{page_text}: {section['summary']}"
                                )
                        status = "ok"
                        return "\n".join(lines)
                except Exception as exc:
                        if isinstance(exc, ToolException):
                                raise
                        raise to_tool_exception(exc) from exc
                finally:
                        _OUTLINE_CALLS.add(1, {"status": status})
                        log.info("document_outline_done", tool="document_outline", status=status,
                                         duration_ms=int((time.perf_counter() - started) * 1000), sections=section_count)

        @tool
        async def delete_document(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
                """Remove one document from the calling user's knowledge base. Use when asked to forget or delete a document.

                Args:
                        doc_name: The document's file name from the knowledge-base list.
                """
                try:
                        user_id = runtime.context.user_id
                        documents = await asyncio.to_thread(kb.documents, user_id)
                        exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
                        if exact is None:
                                raise _unknown_document(doc_name, documents)
                        document = next(item for item in documents if item["filename"] == exact)
                        if not await asyncio.to_thread(kb.delete_document, user_id, document["doc_id"]):
                                raise _unknown_document(doc_name, await asyncio.to_thread(kb.documents, user_id))
                        remaining = await asyncio.to_thread(kb.documents, user_id)
                        return f"Deleted '{exact}'. {len(remaining)} documents left."
                except Exception as exc:
                        if isinstance(exc, ToolException):
                                raise
                        raise to_tool_exception(exc) from exc

        @tool
        async def update_document(doc_name: str, source: str, runtime: ToolRuntime[Ctx]) -> str:
                """Replace a document with a newly uploaded version, retaining its existing file name. Use when asked to update a document.

                Args:
                        doc_name: The existing document's file name from the knowledge-base list.
                        source: The file_id of the newly uploaded version.
                """
                try:
                        user_id = runtime.context.user_id
                        documents = await asyncio.to_thread(kb.documents, user_id)
                        exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
                        if exact is None:
                                raise _unknown_document(doc_name, documents)
                        old_document = next(item for item in documents if item["filename"] == exact)
                        path, _ = await asyncio.to_thread(files.resolve, runtime.context.session_id, source.strip())
                        result, existing = await ingest_file(kb, user_id, path, exact, runtime.config)
                        if existing:
                                return f"'{exact}' is already up to date."
                        await asyncio.to_thread(kb.delete_document, user_id, old_document["doc_id"])
                        return f"Updated '{exact}': {len(result.sections)} sections, {result.n_chunks} chunks."
                except Exception as exc:
                        if isinstance(exc, ToolException):
                                raise
                        raise to_tool_exception(exc) from exc

        for custom_tool in (kb_stats, document_outline, delete_document, update_document):
                custom_tool.handle_tool_error = True
        return [kb_stats, document_outline, delete_document, update_document]
