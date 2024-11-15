from pydantic import BaseModel, Field, validator
from typing import List


class QueueResponse(BaseModel):
    identifier: str
    message: str


class QueueTask(BaseModel):
    identifier: str
    status: str
    task_type: str


class QueueTasksResult(BaseModel):
    tasks: List[QueueTask]
