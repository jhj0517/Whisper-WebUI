import functools
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import wraps
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException
from sqlmodel import SQLModel
from dotenv import load_dotenv

from backend.common.config_loader import read_env


@functools.lru_cache
def init_db():
    db_url = read_env("DB_URL", "sqlite:///backend/records.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    db_instance = init_db()
    return db_instance()


def handle_database_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session = None
        try:
            session = get_db_session()
            kwargs['session'] = session

            return func(*args, **kwargs)
        except Exception as e:
            print(f"Database error has occurred: {e}")
            raise
        finally:
            if session:
                session.close()
    return wrapper
