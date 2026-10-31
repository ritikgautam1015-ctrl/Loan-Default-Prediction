"""Small SQLAlchemy helpers for connecting to PostgreSQL."""

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.config import DATABASE_URL


def verify_database_connection(database_url: str = DATABASE_URL) -> None:
    """Connect to PostgreSQL or raise a readable error."""
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"PostgreSQL connection failed: {exc}. Check that PostgreSQL is running "
            "and DATABASE_URL in .env is correct."
        ) from exc
    finally:
        engine.dispose()
