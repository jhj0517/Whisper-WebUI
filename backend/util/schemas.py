from pydantic import BaseModel, Field, validator


class QueueResponse(BaseModel):
    message: str
