from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class CompareResponse(BaseModel):
    similarity: float
    threshold: float
    same_speaker: bool


class EnrollResponse(BaseModel):
    speaker_id: str
    samples: int


class VerifyResponse(BaseModel):
    speaker_id: str
    similarity: float
    threshold: float
    verified: bool
    samples: int


class SpeakerInfo(BaseModel):
    speaker_id: str
    samples: int


class DeleteResponse(BaseModel):
    speaker_id: str
    deleted: bool


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    duration_seconds: float
