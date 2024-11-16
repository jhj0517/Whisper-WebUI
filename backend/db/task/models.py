# Ported from https://github.com/pavelzbornik/whisperX-FastAPI/blob/main/app/models.py

from pydantic import BaseModel
from typing import List
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Float, JSON, Integer, DateTime
from sqlalchemy.orm import declarative_base


class Task(declarative_base()):
    """
    Table to store tasks information.

    Attributes:
    - id: Unique identifier for each task (Primary Key).
    - uuid: Universally unique identifier for each task.
    - status: Current status of the task.
    - result: JSON data representing the result of the task.
    - file_name: Name of the file associated with the task.
    - task_type: Type/category of the task.
    - duration: Duration of the task execution.
    - error: Error message, if any, associated with the task.
    - created_at: Date and time of creation.
    - updated_at: Date and time of last update.
    """

    __tablename__ = "tasks"
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique identifier for each task (Primary Key)",
    )
    uuid = Column(
        String,
        default=lambda: str(uuid4()),
        comment="Universally unique identifier for each task",
    )
    status = Column(String, comment="Current status of the task")
    result = Column(
        JSON, comment="JSON data representing the result of the task"
    )
    file_name = Column(
        String, comment="Name of the file associated with the task"
    )
    url = Column(String, comment="URL of the file associated with the task")
    audio_duration = Column(Float, comment="Duration of the audio in seconds")
    language = Column(
        String, comment="Language of the file associated with the task"
    )
    task_type = Column(String, comment="Type/category of the task")
    task_params = Column(JSON, comment="Parameters of the task")
    duration = Column(Float, comment="Duration of the task execution")
    error = Column(
        String, comment="Error message, if any, associated with the task"
    )
    created_at = Column(
        DateTime, default=datetime.utcnow, comment="Date and time of creation"
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Date and time of last update",
    )
