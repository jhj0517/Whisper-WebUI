import functools
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import wraps
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException
from dotenv import load_dotenv

from ..common.config_loader import read_env


@functools.lru_cache
def init_db():
    db_url = read_env("DB_URL", "sqlite:///records.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    db_instance = init_db()
    session = db_instance()
    try:
        yield session
    finally:
        session.close()


def handle_database_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as e:
            error_message = f"Database error: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)
    return wrapper
