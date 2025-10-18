from __future__ import annotations

from typing import List, Annotated

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    StringConstraints,
)
from enum import Enum


class EntityType(str, Enum):
    Person = "Person"
    Org = "Org"
    Location = "Location"
    Artifact = "Artifact"
    Concept = "Concept"


class Mention(BaseModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    sent_id: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check_bounds(self):
        # self is the Mention instance
        if self.start >= self.end:
            raise ValueError("Mention start must be < end.")
        return self


class EntityNoId(BaseModel):
    type: EntityType
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    mentions: List[Mention] = Field(default_factory=list)


class EntitiesOutput(BaseModel):
    entities: List[EntityNoId] = Field(default_factory=list)
