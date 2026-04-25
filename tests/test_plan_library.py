from __future__ import annotations

from types import SimpleNamespace

from fin_stock_agent.services import plan_library_service as module
from fin_stock_agent.storage.database import init_db


class _FakeVectorStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, tuple[str, dict]]] = {}

    def upsert(self, collection: str, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self.docs.setdefault(collection, {})[doc_id] = (text, dict(metadata or {}))

    def search(self, collection: str, query_text: str, top_k: int, threshold: float):
        docs = list(self.docs.get(collection, {}).items())[:top_k]
        return [
            SimpleNamespace(doc_id=doc_id, text=text, similarity=max(threshold, 0.95), metadata=meta)
            for doc_id, (text, meta) in docs
        ]

    def delete(self, collection: str, doc_id: str) -> None:
        self.docs.get(collection, {}).pop(doc_id, None)

    def delete_collection(self, collection: str) -> None:
        self.docs.pop(collection, None)


def test_plan_library_save_and_search(monkeypatch) -> None:
    init_db()
    fake_store = _FakeVectorStore()
    monkeypatch.setattr(module, "get_vector_store", lambda: fake_store)
    service = module.PlanLibraryService()
    user_id = "plan-user"

    service.clear_user(user_id)
    service.save_plan(user_id, "分析黄金基金", ["收集新闻", "判断趋势"], quality_score=0.8)
    results = service.search_plans(user_id, "黄金基金分析", top_k=3)

    assert results
    assert results[0].plan_steps == ["收集新闻", "判断趋势"]
    assert results[0].similarity >= 0.82
