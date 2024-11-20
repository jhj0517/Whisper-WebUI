from pydantic import BaseModel, Field


class BGMSeparationResult(BaseModel):
    instrumental_hash: str = Field(..., description="Instrumental file hash")
    vocal_hash: str = Field(..., description="Vocal file hash")
