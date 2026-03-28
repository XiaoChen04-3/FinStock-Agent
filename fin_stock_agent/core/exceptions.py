"""Centralised exception hierarchy for FinStock-Agent."""


class FinStockError(Exception):
    """Base error for FinStock-Agent."""


class TushareRequestError(FinStockError):
    """Tushare API or network failure."""


class AgentRoutingError(FinStockError):
    """Agent routing or complexity-classification failure."""


class MemoryError(FinStockError):
    """Portfolio or conversation memory failure."""
