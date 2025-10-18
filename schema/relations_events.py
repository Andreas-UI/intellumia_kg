from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Span(BaseModel):
    start: int
    end: int
    sent_id: int
    trigger: Optional[Dict[str, int]] = None

    @field_validator("start", "end", "sent_id")
    @classmethod
    def ge_zero(cls, v: int) -> int:
        if v < 0:
            raise ValueError("indices must be >= 0")
        return v

    @field_validator("end")
    @classmethod
    def end_gt_start(cls, v: int, info):
        s = info.data.get("start")
        if s is not None and v <= s:
            raise ValueError("end must be > start")
        return v


class Relation(BaseModel):
    id: str
    type: str
    source_label: str
    head: str
    tail: str
    polarity: str = "asserted"
    modality: str = "factual"
    supports: List[Span]
    place: Optional[Dict[str, str]] = None
    confidence_local: float = Field(ge=0.0, le=1.0, default=0.7)
    rationale: Optional[str] = None


class Event(BaseModel):
    id: str
    type: str
    source_label: str
    participants: List[Dict[str, str]] = Field(default_factory=list)
    status: str = "completed"
    polarity: str = "asserted"
    supports: List[Span]
    # time removed
    place: Optional[Dict[str, str]] = None
    confidence_local: float = Field(ge=0.0, le=1.0, default=0.7)
    rationale: Optional[str] = None


class Output(BaseModel):
    chunk_id: str
    relations: List[Relation] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
