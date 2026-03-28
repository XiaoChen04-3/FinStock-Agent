class FinStockError(Exception):
    """Base error for FinStock-Agent."""


class TushareRequestError(FinStockError):
    """Tushare API or network failure."""
