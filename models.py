from pydantic import BaseModel
from typing import Optional, Literal, List

class TaskRequest(BaseModel):
    taskType: Literal["text", "embedding", "image", "speech-to-text", "text-to-speech", "video"]
    input: str
    constraints: Optional[dict] = None  # e.g., {"fastest": True, "cheapest": False, "hybrid": True}

class TaskResponse(BaseModel):
    provider: str
    taskType: str
    result: str
    cost: float
    latency_ms: int
    status: str
    cached: Optional[bool] = False
    hybrid: Optional[bool] = False
    providers_used: Optional[List[str]] = []
