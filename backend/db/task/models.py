# Ported from https://github.com/pavelzbornik/whisperX-FastAPI/blob/main/app/models.py

from enum import Enum
from pydantic import BaseModel
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
from sqlalchemy.types import Enum as SQLAlchemyEnum
from typing import Any
from sqlmodel import SQLModel, Field, JSON, Column


class ResultType(str, Enum):
    JSON = "json"
    FILEPATH = "filepath"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    PAUSED = "paused"
    RETRYING = "retrying"

    def __str__(self):
        return self.value


class TaskType(str, Enum):
    TRANSCRIPTION = "transcription"
    VAD = "vad"
    BGM_SEPARATION = "bgm_separation"

    def __str__(self):
        return self.value


class TaskStatusResponse(BaseModel):
    """`TaskStatusResponse` is a wrapper class that hides sensitive information from `Task`"""
    identifier: str = Field(..., description="Unique identifier for the queued task that can be used for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    task_type: Optional[TaskType] = Field(
        default=None,
        description="Type/category of the task"
    )
    result_type: Optional[ResultType] = Field(
        default=ResultType.JSON,
        description="Result type whether it's a filepath or JSON"
    )
    result: Optional[Any] = Field(
        default=None,
        description="JSON data representing the result of the task"
    )
    task_params: Optional[dict] = Field(
        default=None,
        description="Parameters of the task"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message, if any, associated with the task"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of the task execution"
    )


class Task(SQLModel, table=True):
    """
    Table to store tasks information.

    Attributes:
    - id: Unique identifier for each task (Primary Key).
    - uuid: Universally unique identifier for each task.
    - status: Current status of the task.
    - result: JSON data representing the result of the task.
    - result_type: Type of the data whether it is normal JSON data or filepath.
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
    status: Optional[TaskStatus] = Field(
        default=None,
        sa_column=Field(sa_column=SQLAlchemyEnum(TaskStatus)),
        description="Current status of the task",
    )
    result: Optional[dict] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="JSON data representing the result of the task"
    )
    result_type: Optional[ResultType] = Field(
        default=ResultType.JSON,
        sa_column=Field(sa_column=SQLAlchemyEnum(ResultType)),
        description="Result type whether it's a filepath or JSON"
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
    task_type: Optional[TaskType] = Field(
        default=None,
        sa_column=Field(sa_column=SQLAlchemyEnum(TaskType)),
        description="Type/category of the task"
    )
    task_params: Optional[dict] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
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

    def to_response(self) -> "TaskStatusResponse":
        return TaskStatusResponse(
            identifier=self.uuid,
            status=self.status,
            task_type=self.task_type,
            result_type=self.result_type,
            result=self.result,
            task_params=self.task_params,
            error=self.error,
            duration=self.duration
        )


class TasksResult(BaseModel):
    tasks: List[Task]

