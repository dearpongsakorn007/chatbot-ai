from typing import Optional
from pydantic import BaseModel


class LineSource(BaseModel):
    userId: Optional[str] = None
    type: Optional[str] = None


class LineMessage(BaseModel):
    type: str
    text: Optional[str] = None


class LineEvent(BaseModel):
    type: str
    replyToken: Optional[str] = None
    source: Optional[LineSource] = None
    message: Optional[LineMessage] = None


class LineWebhookPayload(BaseModel):
    destination: Optional[str] = None
    events: list[LineEvent] = []


class RetrievedChunk(BaseModel):
    content: str
    verified_evidence: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    reference: Optional[str] = None
    image_url: Optional[str] = None
    preview_image_url: Optional[str] = None
