from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from fin_stock_agent.core.config import get_config
from fin_stock_agent.memory.vector_store import get_vector_store
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import PlanLibraryORM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanCandidate:
    plan_steps: list[str]
    quality_score: float
    similarity: float
    question_text: str
    chroma_doc_id: str


class PlanLibraryService:
    def __init__(self) -> None:
        self._cfg = get_config().memory.plan_library

    def save_plan(self, user_id: str, question_text: str, plan_steps: list[str], quality_score: float) -> None:
        if not question_text.strip() or not plan_steps:
            return
        if quality_score < self._cfg.min_quality_score:
            return
        collection = self._collection_name(user_id)
        doc_id = f"plan_{uuid.uuid4().hex[:24]}"
        self._evict_if_needed(user_id)
        get_vector_store().upsert(collection, doc_id, question_text, {"user_id": user_id, "kind": "plan"})
        with get_session() as session:
            session.add(
                PlanLibraryORM(
                    user_id=user_id,
                    question_text=question_text[:1000],
                    chroma_doc_id=doc_id,
                    plan_json=json.dumps(plan_steps, ensure_ascii=False),
                    quality_score=max(0.0, min(1.0, quality_score)),
                    use_count=0,
                )
            )

    def search_plans(self, user_id: str, question_text: str, top_k: int = 3) -> list[PlanCandidate]:
        if not question_text.strip():
            return []
        try:
            results = get_vector_store().search(
                self._collection_name(user_id),
                question_text,
                top_k=top_k,
                threshold=self._cfg.reference_threshold,
            )
        except Exception as exc:
            logger.warning("Plan library search failed: %s", exc)
            return []
        if not results:
            return []
        ids = [item.doc_id for item in results]
        with get_session() as session:
            rows = session.execute(
                select(PlanLibraryORM).where(
                    PlanLibraryORM.user_id == user_id,
                    PlanLibraryORM.chroma_doc_id.in_(ids),
                )
            ).scalars()
            mapping = {row.chroma_doc_id: row for row in rows}
            candidates: list[PlanCandidate] = []
            for result in results:
                row = mapping.get(result.doc_id)
                if row is None:
                    continue
                row.use_count = int(row.use_count or 0) + 1
                row.last_used_at = datetime.utcnow()
                try:
                    plan_steps = [str(item) for item in json.loads(row.plan_json)]
                except Exception:
                    plan_steps = []
                candidates.append(
                    PlanCandidate(
                        plan_steps=plan_steps,
                        quality_score=float(row.quality_score or 0.0),
                        similarity=result.similarity,
                        question_text=row.question_text,
                        chroma_doc_id=row.chroma_doc_id,
                    )
                )
            return candidates

    def list_plans(self, user_id: str, limit: int = 20) -> list[dict]:
        with get_session() as session:
            rows = session.execute(
                select(PlanLibraryORM)
                .where(PlanLibraryORM.user_id == user_id)
                .order_by(PlanLibraryORM.last_used_at.desc().nullslast(), PlanLibraryORM.created_at.desc())
                .limit(limit)
            ).scalars()
            return [
                {
                    "question_text": row.question_text,
                    "plan_steps": json.loads(row.plan_json),
                    "quality_score": float(row.quality_score or 0.0),
                    "use_count": int(row.use_count or 0),
                    "created_at": row.created_at.isoformat() if row.created_at else "",
                    "last_used_at": row.last_used_at.isoformat() if row.last_used_at else "",
                }
                for row in rows
            ]

    def clear_user(self, user_id: str) -> None:
        collection = self._collection_name(user_id)
        with get_session() as session:
            rows = session.execute(select(PlanLibraryORM).where(PlanLibraryORM.user_id == user_id)).scalars().all()
            for row in rows:
                try:
                    get_vector_store().delete(collection, row.chroma_doc_id)
                except Exception:
                    logger.debug("Failed deleting plan vector %s", row.chroma_doc_id)
                session.delete(row)
        try:
            get_vector_store().delete_collection(collection)
        except Exception:
            logger.debug("Ignoring plan collection delete failure for %s", collection)

    def _evict_if_needed(self, user_id: str) -> None:
        with get_session() as session:
            rows = session.execute(
                select(PlanLibraryORM)
                .where(PlanLibraryORM.user_id == user_id)
                .order_by(PlanLibraryORM.use_count.asc(), PlanLibraryORM.created_at.asc())
            ).scalars().all()
            overflow = max(0, len(rows) - self._cfg.max_records_per_user + 1)
            collection = self._collection_name(user_id)
            for row in rows[:overflow]:
                try:
                    get_vector_store().delete(collection, row.chroma_doc_id)
                except Exception:
                    logger.debug("Failed evicting plan vector %s", row.chroma_doc_id)
                session.delete(row)

    @staticmethod
    def _collection_name(user_id: str) -> str:
        return f"plan_library_{user_id}"
