from __future__ import annotations

from app_streamlit import _extract_market_report_part, _normalize_top_news


def test_extract_market_report_part_splits_before_top10_section() -> None:
    market_context = (
        "**当日要点**  市场风险偏好回升，科技成长方向更活跃。\n\n"
        "**重点十条**\n"
        "1. 新闻A\n"
        "2. 新闻B"
    )

    result = _extract_market_report_part(market_context)

    assert result == "**当日要点**  市场风险偏好回升，科技成长方向更活跃。"


def test_normalize_top_news_limits_and_keeps_structured_fields() -> None:
    items = [
        {
            "rank": 3,
            "title": "黄金价格再度走强",
            "source": "cls",
            "reason": "避险情绪升温",
            "impact": 5,
            "time": "2026-04-12T09:30:00",
        },
        {"title": ""},
    ] + [{"title": f"新闻{i}"} for i in range(2, 15)]

    normalized = _normalize_top_news(items)

    assert len(normalized) == 10
    assert normalized[0]["rank"] == 3
    assert normalized[0]["title"] == "黄金价格再度走强"
    assert normalized[0]["source"] == "cls"
    assert normalized[0]["reason"] == "避险情绪升温"
    assert normalized[0]["impact"] == 5
