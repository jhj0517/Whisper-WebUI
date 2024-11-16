from pydantic import BaseModel, Field, validator
from typing import List


class QueueResponse(BaseModel):
    identifier: str
    message: str


