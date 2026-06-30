"""
backend/core/database.py
Async SQLAlchemy engine + session factory.
Supports PostgreSQL (production) and SQLite (development fallback).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _build_engine():
    db_url = settings.effective_database_url
    if settings.use_sqlite:
        engine = create_async_engine(
            db_url,
            echo=settings.debug,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_async_engine(
            db_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    logger.info("Database engine created: %s", db_url.split("@")[-1] if "@" in db_url else db_url)
    return engine


engine = _build_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


_db_initialized = False


async def init_db() -> None:
    """Create all tables (idempotent)."""
    # Import models to register them with Base.metadata
    import backend.models.orm  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")
    
    # Create required directories (lifespan doesn't run on Vercel serverless)
    import os
    for d in [settings.datasets_dir, settings.model_dir, settings.reports_dir, settings.mast_cache_dir]:
        os.makedirs(d, exist_ok=True)


async def ensure_db_initialized():
    global _db_initialized
    if not _db_initialized:
        await init_db()
        _db_initialized = True


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    await engine.dispose()
    logger.info("Database engine disposed.")


@asynccontextmanager
async def get_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    """Context-manager for a single database session."""
    await ensure_db_initialized()
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    await ensure_db_initialized()
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
