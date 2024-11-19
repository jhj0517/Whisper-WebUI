from pydantic import BaseModel, Field


class BGMSeparationResult(BaseModel):
    instrumental_path: str = Field(..., description="Instrumental file path")
    vocal_path: str = Field(..., description="Vocal file path")
