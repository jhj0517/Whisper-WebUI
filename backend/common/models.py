from pydantic import BaseModel, Field, validator
from typing import List, Any, Optional
from backend.db.task.models import TaskStatus, ResultType, TaskType


class QueueResponse(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the queued task that can be used for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    message: str = Field(..., description="Message providing additional information about the task")


class TaskStatusResponse(BaseModel):
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
    result: Optional[dict] = Field(
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


class Response(BaseModel):
    identifier: str
    message: str


class Metadata(BaseModel):
    task_type: str
    task_params: Optional[dict]
    language: Optional[str]
    file_name: Optional[str]
    url: Optional[str]
    duration: Optional[float]
    audio_duration: Optional[float] = None


class Result(BaseModel):
    status: str
    result: Any
    metadata: Metadata
    error: Optional[str]
