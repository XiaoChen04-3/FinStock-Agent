"""Datetime utility tool – lets the Agent resolve relative time expressions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """
    返回当前日期、时间、星期及近 7 天日期列表（YYYYMMDD）。
    当用户提到「今天」「最近几天」「本周」「昨天」等相对时间时，必须先调用此工具
    获取精确日期，再传给行情/指数查询工具，避免日期计算错误。
    """
    now = datetime.now()
    recent_7 = [(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    recent_30 = [(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(30)]
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return json.dumps(
        {
            "today": now.strftime("%Y%m%d"),
            "today_display": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": weekday_names[now.weekday()],
            "is_weekday": now.weekday() < 5,
            "last_7_days": recent_7,
            "last_30_days": recent_30,
            "note": (
                "A 股交易日为周一至周五（国家法定节假日除外）。"
                "查询日线行情时请用 last_7_days / last_30_days 中的日期，"
                "如接口返回空则说明该日不是交易日，尝试前一天。"
            ),
        },
        ensure_ascii=False,
    )


def get_datetime_tools():
    return [get_current_datetime]
