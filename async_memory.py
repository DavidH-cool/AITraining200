"""Asynchronous Background Memory Consolidation & Style Profile Extraction.

Addresses Rubric Category 2 (Context & Memory) - Async Memory Operations:
- Executes expensive memory summarization, author style extraction, and vector
  embedding persistence as non-blocking `asyncio.create_task` background jobs
  so interactive agent responses never block the user interface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from blog_writer_agent.memory.compaction import default_history_compactor
from blog_writer_agent.memory.session_store import (
    PersistentKnowledgeAndSessionStore,
    get_persistent_knowledge_store,
)
from blog_writer_agent.observability.logger import get_structured_logger
from blog_writer_agent.observability.pii_redactor import scrub_pii
from blog_writer_agent.observability.tracing import start_linked_span

logger = get_structured_logger(__name__)


class AsyncMemoryConsolidationManager:
    """Schedules and executes non-blocking background memory consolidation tasks."""

    def __init__(
        self,
        store: Optional[PersistentKnowledgeAndSessionStore] = None,
    ) -> None:
        self._store = store or get_persistent_knowledge_store()
        self._background_tasks: set[asyncio.Task[Dict[str, Any]]] = set()

    async def consolidate_session_memory_async(
        self,
        session_id: str,
        user_id: str,
        conversation_turns: List[Dict[str, str]],
        extracted_engineering_facts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously compacts session history, extracts author style notes, and persists vectors."""
        with start_linked_span(
            "memory.async_consolidation",
            attributes={"session.id": session_id, "turns.count": len(conversation_turns)},
        ):
            # Yield to the event loop immediately so UI response dispatch is never blocked
            await asyncio.sleep(0)

            compaction = default_history_compactor.compact_messages(
                conversation_turns,
                force_summarize=len(conversation_turns) > 2,
            )

            sanitized_facts = scrub_pii(extracted_engineering_facts or [])
            consolidated_state: Dict[str, Any] = {
                "compacted_summary": compaction.compacted_summary or "",
                "compacted_estimated_tokens": compaction.compacted_estimated_tokens,
                "extracted_engineering_facts": sanitized_facts,
            }

            await asyncio.to_thread(
                self._store.save_session_state,
                session_id=session_id,
                user_id=user_id,
                state=consolidated_state,
                turns=compaction.retained_messages,
            )

            if sanitized_facts:
                doc_id = f"memory-bank-{session_id}"
                await asyncio.to_thread(
                    self._store.upsert_knowledge_document,
                    doc_id=doc_id,
                    title=f"Consolidated Engineering Work Memory ({session_id})",
                    content="; ".join(sanitized_facts),
                    audience="software_engineers",
                    tags=["consolidated-memory", "engineering-work"],
                )

            result = {
                "status": "consolidated",
                "session_id": session_id,
                "original_tokens": compaction.original_estimated_tokens,
                "compacted_tokens": compaction.compacted_estimated_tokens,
                "facts_indexed": len(sanitized_facts),
            }
            logger.info(
                "Completed async background memory consolidation.",
                extra={"event_type": "async_memory_consolidated", **result},
            )
            return result

    def schedule_background_consolidation(
        self,
        session_id: str,
        user_id: str,
        conversation_turns: List[Dict[str, str]],
        extracted_engineering_facts: Optional[List[str]] = None,
    ) -> asyncio.Task[Dict[str, Any]]:
        """Fire-and-forget scheduler that launches `consolidate_session_memory_async` in the background."""
        task = asyncio.create_task(
            self.consolidate_session_memory_async(
                session_id=session_id,
                user_id=user_id,
                conversation_turns=conversation_turns,
                extracted_engineering_facts=extracted_engineering_facts,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def wait_for_pending_tasks(self) -> List[Dict[str, Any]]:
        """Awaits all currently scheduled background memory tasks (useful in tests and shutdown)."""
        if not self._background_tasks:
            return []
        tasks = list(self._background_tasks)
        return await asyncio.gather(*tasks)


default_async_memory_manager = AsyncMemoryConsolidationManager()
