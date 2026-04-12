from __future__ import annotations

import re

from fin_stock_agent.memory.profile_memory import MemoryEvent, MemoryExtractionResult, UserProfileMemory

_RISK_PATTERNS = [
    ("进取", ("进取", "激进", "高风险", "高波动也可以")),
    ("平衡", ("平衡", "均衡")),
    ("稳健", ("稳健", "中低风险")),
    ("保守", ("保守", "低风险", "不想大波动", "不接受大回撤")),
]

_HORIZON_PATTERNS = [
    ("短线", ("短线", "短期", "快进快出")),
    ("中期", ("中期", "半年到一年")),
    ("长期", ("长期", "长线", "定投", "三年以上")),
]

_ASSET_KEYWORDS = [
    "ETF",
    "LOF",
    "宽基",
    "指数基金",
    "主动基金",
    "债基",
    "债券基金",
    "货币基金",
    "QDII",
    "REITs",
    "股票",
    "基金",
]

_THEME_KEYWORDS = [
    "黄金",
    "红利",
    "医药",
    "创新药",
    "白酒",
    "消费",
    "半导体",
    "芯片",
    "AI",
    "人工智能",
    "算力",
    "机器人",
    "新能源",
    "光伏",
    "军工",
    "银行",
    "券商",
    "港股",
    "美股",
    "债券",
    "煤炭",
    "有色",
]

_ANSWER_STYLE_RULES = [
    ("简洁", ("简洁", "精简", "简单一点", "短一点")),
    ("结论先行", ("先给结论", "结论先行", "先说结论")),
    ("表格", ("表格", "表格化")),
    ("风险提示", ("风险", "列出风险", "提醒风险")),
]

_CONSTRAINT_PATTERNS = [
    "不要推荐具体股票",
    "不想追高",
    "不要追高",
    "不接受大回撤",
    "不碰个股",
    "不碰股票",
    "不想高波动",
]


def _contains_any(text: str, phrases: tuple[str, ...] | list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_by_patterns(text: str, rules: list[tuple[str, tuple[str, ...]]]) -> str:
    for label, patterns in rules:
        if _contains_any(text, patterns):
            return label
    return ""


def _extract_list(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def _extract_preferred_assets(text: str) -> list[str]:
    positive_cues = ("偏好", "喜欢", "主要买", "只看", "配置", "持有")
    if not _contains_any(text, positive_cues):
        return []
    items: list[str] = []
    for keyword in _ASSET_KEYWORDS:
        if keyword not in text:
            continue
        if f"不碰{keyword}" in text or f"不看{keyword}" in text or f"不要{keyword}" in text:
            continue
        items.append(keyword)
    return items


def _extract_watchlist(text: str) -> list[str]:
    found = re.findall(r"\b\d{6}\.(?:SH|SZ|OF)\b", text.upper())
    items = list(found)
    for theme in _extract_list(text, _THEME_KEYWORDS):
        if any(trigger in text for trigger in ("关注", "跟踪", "观察", "盯着", "留意")):
            items.append(theme)
    return _unique_keep_order(items)[:8]


def _merge_list(existing: list[str], incoming: list[str]) -> list[str]:
    return _unique_keep_order(list(existing) + list(incoming))


def extract_memory_updates(
    question: str,
    answer: str,
    *,
    existing_profile: UserProfileMemory | None = None,
) -> MemoryExtractionResult:
    profile = existing_profile or UserProfileMemory()
    text = " ".join(part.strip() for part in [question or "", answer or ""] if part.strip())

    updates: dict = {}
    events: list[MemoryEvent] = []

    risk_level = _extract_by_patterns(text, _RISK_PATTERNS)
    if risk_level and risk_level != profile.risk_level:
        updates["risk_level"] = risk_level
        events.append(
            MemoryEvent(
                event_type="risk_preference",
                summary=f"用户风险偏好更新为{risk_level}",
                payload={"risk_level": risk_level},
                confidence=0.9,
                source_text=question,
            )
        )

    investment_horizon = _extract_by_patterns(text, _HORIZON_PATTERNS)
    if investment_horizon and investment_horizon != profile.investment_horizon:
        updates["investment_horizon"] = investment_horizon
        events.append(
            MemoryEvent(
                event_type="investment_horizon",
                summary=f"用户投资期限偏好更新为{investment_horizon}",
                payload={"investment_horizon": investment_horizon},
                confidence=0.82,
                source_text=question,
            )
        )

    preferred_assets = _extract_preferred_assets(text)
    if preferred_assets:
        merged = _merge_list(profile.preferred_assets, preferred_assets)
        if merged != profile.preferred_assets:
            updates["preferred_assets"] = merged
            events.append(
                MemoryEvent(
                    event_type="preferred_assets",
                    summary=f"用户偏好资产类型：{', '.join(preferred_assets[:4])}",
                    payload={"preferred_assets": preferred_assets[:8]},
                    confidence=0.76,
                    source_text=question,
                )
            )

    disliked_assets: list[str] = []
    if "不碰股票" in text or "不碰个股" in text:
        disliked_assets.append("股票")
    if "不碰基金" in text:
        disliked_assets.append("基金")
    if disliked_assets:
        merged = _merge_list(profile.disliked_assets, disliked_assets)
        if merged != profile.disliked_assets:
            updates["disliked_assets"] = merged
            events.append(
                MemoryEvent(
                    event_type="disliked_assets",
                    summary=f"用户明确回避：{', '.join(disliked_assets)}",
                    payload={"disliked_assets": disliked_assets},
                    confidence=0.88,
                    source_text=question,
                )
            )

    focus_themes = _extract_list(text, _THEME_KEYWORDS)
    if focus_themes:
        merged = _merge_list(profile.focus_themes, focus_themes)
        if merged != profile.focus_themes:
            updates["focus_themes"] = merged
            events.append(
                MemoryEvent(
                    event_type="focus_themes",
                    summary=f"用户关注主题：{', '.join(focus_themes[:5])}",
                    payload={"focus_themes": focus_themes[:8]},
                    confidence=0.8,
                    source_text=question,
                )
            )

    answer_style = [label for label, patterns in _ANSWER_STYLE_RULES if _contains_any(text, patterns)]
    if answer_style:
        merged = _merge_list(profile.answer_style, answer_style)
        if merged != profile.answer_style:
            updates["answer_style"] = merged
            events.append(
                MemoryEvent(
                    event_type="answer_style",
                    summary=f"用户回答偏好：{', '.join(answer_style)}",
                    payload={"answer_style": answer_style},
                    confidence=0.84,
                    source_text=question,
                )
            )

    constraints = [pattern for pattern in _CONSTRAINT_PATTERNS if pattern in text]
    if constraints:
        merged = _merge_list(profile.decision_constraints, constraints)
        if merged != profile.decision_constraints:
            updates["decision_constraints"] = merged
            events.append(
                MemoryEvent(
                    event_type="decision_constraints",
                    summary=f"用户约束更新：{', '.join(constraints[:4])}",
                    payload={"decision_constraints": constraints[:8]},
                    confidence=0.9,
                    source_text=question,
                )
            )

    watchlist = _extract_watchlist(text)
    if watchlist:
        merged = _merge_list(profile.watchlist, watchlist)
        if merged != profile.watchlist:
            updates["watchlist"] = merged
            events.append(
                MemoryEvent(
                    event_type="watchlist",
                    summary=f"用户加入观察名单：{', '.join(watchlist[:5])}",
                    payload={"watchlist": watchlist[:8]},
                    confidence=0.72,
                    source_text=question,
                )
            )

    return MemoryExtractionResult(profile_updates=updates, events=events)
