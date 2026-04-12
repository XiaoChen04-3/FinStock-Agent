from fin_stock_agent.tools.fund import get_fund_tools
from fin_stock_agent.tools.index import get_index_tools
from fin_stock_agent.tools.macro import get_macro_tools
from fin_stock_agent.tools.market import get_market_tools
from fin_stock_agent.tools.news import get_news_tools
from fin_stock_agent.tools.portfolio import get_portfolio_tools
from fin_stock_agent.tools.system import get_system_tools
from fin_stock_agent.tools.technical import get_technical_tools
from fin_stock_agent.utils.datetime_utils import get_datetime_tools


def get_all_tools() -> list:
    return [
        *get_datetime_tools(),
        *get_market_tools(),
        *get_fund_tools(),
        *get_index_tools(),
        *get_technical_tools(),
        *get_macro_tools(),
        *get_portfolio_tools(),
        *get_news_tools(),
        *get_system_tools(),
    ]


__all__ = ["get_all_tools"]
