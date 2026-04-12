from __future__ import annotations

import logging
from threading import Lock

_LOCK = Lock()
_CONFIGURED = False


def configure_application_logging() -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return

        logging.getLogger("fin_stock_agent").setLevel(logging.ERROR)
        logging.getLogger("apscheduler").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

        _CONFIGURED = True
