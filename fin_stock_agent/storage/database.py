from __future__ import annotations

import logging
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from fin_stock_agent.core.settings import settings

logger = logging.getLogger(__name__)


def _make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    created_engine = create_engine(database_url, future=True, connect_args=connect_args)
    created_session = sessionmaker(bind=created_engine, autoflush=False, autocommit=False, future=True)
    return created_engine, created_session


engine, SessionLocal = _make_engine(settings.database_url)


def _fallback_runtime_database() -> str:
    return f"sqlite:///{(settings.project_root / 'finstock_runtime.db').as_posix()}"


def _apply_schema_migrations(eng) -> None:
    """Add columns that exist in ORM models but are absent from the live database.

    SQLite only supports ALTER TABLE … ADD COLUMN, so this handles the common case
    of new nullable / default-carrying columns added to models after the DB was created.
    Existing columns are left untouched; failures are logged but never fatal.
    """
    from fin_stock_agent.storage.models import Base

    with eng.connect() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                result = conn.execute(sa.text(f"PRAGMA table_info({table.name})"))
                existing_cols = {row[1] for row in result.fetchall()}
            except Exception as exc:
                logger.warning("Schema migration: could not inspect table %s: %s", table.name, exc)
                continue

            for col in table.columns:
                if col.name in existing_cols:
                    continue
                col_type = col.type.compile(eng.dialect)
                sql = f"ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}"
                # Add a literal DEFAULT clause only for scalar defaults (not callables / lambdas).
                if col.default is not None and getattr(col.default, "is_scalar", False):
                    val = col.default.arg
                    if isinstance(val, bool):
                        sql += f" DEFAULT {int(val)}"
                    elif isinstance(val, (int, float)):
                        sql += f" DEFAULT {val}"
                    elif isinstance(val, str):
                        sql += f" DEFAULT '{val}'"
                try:
                    conn.execute(sa.text(sql))
                    conn.commit()
                    logger.info("Schema migration: added column %s.%s (%s)", table.name, col.name, col_type)
                except Exception as exc:
                    logger.warning(
                        "Schema migration: could not add column %s.%s: %s",
                        table.name, col.name, exc,
                    )


def init_db() -> None:
    from fin_stock_agent.storage.models import Base

    global engine, SessionLocal
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        message = str(exc).lower()
        default_url = f"sqlite:///{(settings.data_dir / ('finstock_test.db' if settings._is_pytest_runtime() else 'finstock.db')).as_posix()}"
        recoverable = "readonly" in message or "unable to open database file" in message
        if not recoverable or settings.database_url != default_url:
            raise
        runtime_url = _fallback_runtime_database()
        engine, SessionLocal = _make_engine(runtime_url)
        Base.metadata.create_all(bind=engine)

    _apply_schema_migrations(engine)


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
