from pydantic import BaseModel, Field, validator
from typing import List, Any, Optional
from backend.db.task.models import TaskStatus


class QueueResponse(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the queued task that can be used for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    message: str = Field(..., description="Message providing additional information about the task")


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
