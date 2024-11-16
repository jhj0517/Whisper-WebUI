from pydantic import BaseModel, Field, validator
from typing import List
from enum import Enum


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


class QueueTask(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the task")
    status: TaskStatus = Field(..., description="Current status of the task")
    task_type: str = Field(..., description="Type or category of the task")


class QueueTasksResult(BaseModel):
    tasks: List[QueueTask]
