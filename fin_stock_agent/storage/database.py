from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from fin_stock_agent.core.settings import settings


def _make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    created_engine = create_engine(database_url, future=True, connect_args=connect_args)
    created_session = sessionmaker(bind=created_engine, autoflush=False, autocommit=False, future=True)
    return created_engine, created_session


engine, SessionLocal = _make_engine(settings.database_url)


def _fallback_runtime_database() -> str:
    return f"sqlite:///{(settings.project_root / 'finstock_runtime.db').as_posix()}"


def init_db() -> None:
    from fin_stock_agent.storage.models import Base

    global engine, SessionLocal
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        message = str(exc).lower()
        if "readonly" not in message or settings.database_url != f"sqlite:///{(settings.data_dir / ('finstock_test.db' if settings._is_pytest_runtime() else 'finstock.db')).as_posix()}":
            raise
        runtime_url = _fallback_runtime_database()
        engine, SessionLocal = _make_engine(runtime_url)
        Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
