"""Verify TUSHARE_TOKEN connectivity. Usage: python verify_tushare.py"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    from fin_stock_agent.core.settings import settings

    if not settings.tushare_token:
        print("错误: 未设置 TUSHARE_TOKEN，请在 .env 中配置。")
        return 1

    from fin_stock_agent.utils.tushare_client import TushareClient

    try:
        c = TushareClient(token=settings.tushare_token, cache_enabled=False)
        df = c.call("stock_basic", list_status="L", fields="ts_code,name")
    except Exception as e:
        print("调用失败:", e)
        return 2

    if df is None or df.empty:
        print("stock_basic 返回空，请检查 token/权限")
        return 3

    print("连通成功，总行数:", len(df))
    print(df.head(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
