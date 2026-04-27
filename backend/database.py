"""
BioSpark-Light database — SQLite only.

The DB file lives in the user's data dir (see config.USER_DATA_DIR) so it
survives app reinstalls, doesn't require write access to Program Files,
and follows OS conventions.
"""

import logging
import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import DB_FILE

log = logging.getLogger("biospark.db")

# Allow override (e.g. tests) but default to the per-user SQLite file.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{DB_FILE.as_posix()}",
)

_driver = DATABASE_URL.split("://", 1)[0]
log.warning("DB engine initialized: driver=%s file=%s", _driver, DB_FILE)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables. Called on app startup."""
    # Import model modules so their tables register with Base.metadata
    from backend.models import training_history  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.warning("DB tables created/verified")


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a database session."""
    async with async_session() as session:
        yield session
