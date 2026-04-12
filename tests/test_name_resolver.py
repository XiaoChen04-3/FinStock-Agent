from sqlalchemy import delete

from fin_stock_agent.init.name_resolver import NameResolver
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import FundLookupRecord, IndexLookupRecord


def setup_module() -> None:
    init_db()
    with get_session() as session:
        session.execute(delete(FundLookupRecord))
        session.execute(delete(IndexLookupRecord))
        session.add_all(
            [
                FundLookupRecord(ts_code="161725.OF", name="招商中证白酒", fund_type="fund", status="L", market="O"),
                FundLookupRecord(ts_code="110022.OF", name="易方达消费", fund_type="fund", status="L", market="O"),
                IndexLookupRecord(ts_code="000300.SH", name="沪深300", market="SSE", category="broad"),
            ]
        )


def test_resolve_fund_and_index() -> None:
    resolver = NameResolver()
    assert resolver.resolve_fund("招商白酒") == "161725.OF"
    assert resolver.resolve_index("沪深300") == "000300.SH"


def test_search_returns_ranked_candidates() -> None:
    resolver = NameResolver()
    results = resolver.search("消费", top_k=3)
    assert results
    assert results[0]["ts_code"] == "110022.OF"
