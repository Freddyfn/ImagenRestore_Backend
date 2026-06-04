from pydantic import BaseModel
from typing import List

class IterationData(BaseModel):
    iteration: int
    error: float

class RestorationMetrics(BaseModel):
    time_seconds: float
    final_error: float
    iterations: int
    method: str

class ImageRestorationResult(BaseModel):
    filename: str
    status: str
    message: str
    restored_image_base64: str
    metrics: RestorationMetrics
    chart_data: List[IterationData]
