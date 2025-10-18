from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Mention(BaseModel):
    start: int
    end: int
    sent_id: int

    @field_validator("start", "end", "sent_id")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("mention indices must be >= 0")
        return v

    @field_validator("end")
    @classmethod
    def end_gt_start(cls, v: int, info):
        start = info.data.get("start", None)
        if start is not None and v <= start:
            raise ValueError("end must be > start")
        return v


class P1Entity(BaseModel):
    id: str
    type: str  # "Person" | "Org" | "Location" | "Artifact" | "Concept"
    name: str
    mentions: List[Mention] = Field(default_factory=list)


class P1ChunkEntities(BaseModel):
    chunk_id: str
    entities: List[P1Entity]


class CorefDecision(BaseModel):
    same: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class CanonicalEntity(BaseModel):
    id: str  # canonical ID (e.g., "ent_0")
    type: str
    name: str  # canonical display name
    aliases: List[str] = Field(default_factory=list)
    members: List[str] = Field(
        default_factory=list
    )  # list of mention-level IDs (chunk_id + entity.id)


class CorefResult(BaseModel):
    entities_canonical: List[CanonicalEntity]
    coref_map: Dict[str, str]  # mention_id -> canonical_id
