from pydantic import BaseModel, Field


class ImageSize(BaseModel):
    width: int
    height: int


class CropInfo(BaseModel):
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    aspect_ratio: float


class UploadedImageInfo(BaseModel):
    filename: str
    source_format: str
    bytes: int
    original_size: ImageSize
    analyzed_crop: CropInfo


class EvidenceItem(BaseModel):
    id: str
    kind: str
    role: str
    bbox: list[int]
    text: str | None = None
    importance: float
    reason: str


class TaskResult(BaseModel):
    task: str
    label: str | None
    llm_label: str | None = None
    model: str | None
    error: str | None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidences: dict[str, float] | None = None


class AnalyzeResponse(BaseModel):
    image: UploadedImageInfo
    bias: TaskResult
    factuality: TaskResult
    screenshot_b64: str | None = None


class ReasonRequest(BaseModel):
    screenshot_b64: str
    factuality_label: str | None = None
    bias_label: str | None = None
    evidence: list[dict] = Field(default_factory=list)


class ReasonResponse(BaseModel):
    reasoning: str


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    models: dict
    game_items: int


class GameItemResponse(BaseModel):
    id: str
    image_url: str


class GameAnswerRequest(BaseModel):
    id: str
    bias: str
    factuality: str


class GameAnswerResult(BaseModel):
    id: str
    bias: dict
    factuality: dict
