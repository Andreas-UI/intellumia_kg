from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TraitEvidenceItem(BaseModel):
    span: List[int] = Field(min_length=2, max_length=2)
    text: str
    hedges: List[str] = Field(default_factory=list)
    neg: bool = False

    @field_validator("span")
    @classmethod
    def span_ok(cls, v: List[int]) -> List[int]:
        s, e = v
        if s < 0 or e <= s:
            raise ValueError("span must be [start,end] with start>=0 and end>start")
        return v


class Output(BaseModel):
    person: str
    trait_probs: Dict[str, float] = Field(default_factory=dict)
    evidence: List[TraitEvidenceItem] = Field(default_factory=list)
    rationale: Optional[str] = None
