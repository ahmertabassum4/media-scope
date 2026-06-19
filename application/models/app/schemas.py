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


class AnalyzeResponse(BaseModel):
    image: UploadedImageInfo
    bias: TaskResult
    factuality: TaskResult


class ModelsHealthResponse(BaseModel):
    status: str = Field(default="ok")
    openrouter_configured: bool
    bias_openrouter_model: str
    factuality_openrouter_model: str
    bias_pipeline_loaded: bool
    factuality_pipeline_loaded: bool
