# Ported from https://github.com/pavelzbornik/whisperX-FastAPI/blob/main/app/models.py

from typing import Optional
from uuid import uuid4
from datetime import datetime
from sqlmodel import SQLModel, Field, JSON


class Task(SQLModel, table=True):
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

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Unique identifier for each task (Primary Key)"
    )
    uuid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Universally unique identifier for each task"
    )
    status: Optional[str] = Field(
        default=None,
        description="Current status of the task"
    )
    result: Optional[dict] = Field(
        default=None,
        sa_column=JSON,
        description="JSON data representing the result of the task"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Name of the file associated with the task"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL of the file associated with the task"
    )
    audio_duration: Optional[float] = Field(
        default=None,
        description="Duration of the audio in seconds"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language of the file associated with the task"
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Type/category of the task"
    )
    task_params: Optional[dict] = Field(
        default=None,
        sa_column=JSON,
        description="Parameters of the task"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of the task execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message, if any, associated with the task"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date and time of creation"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"onupdate": datetime.utcnow},
        description="Date and time of last update"
    )
