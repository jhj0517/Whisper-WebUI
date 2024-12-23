from pydantic import BaseModel, Field, validator
from typing import List, Any, Optional
from backend.db.task.models import TaskStatus, ResultType, TaskType


class QueueResponse(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the queued task that can be used for tracking")
    status: TaskStatus = Field(..., description="Current status of the task")
    message: str = Field(..., description="Message providing additional information about the task")


class Response(BaseModel):
    identifier: str
    message: str
