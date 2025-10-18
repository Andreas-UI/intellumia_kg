from __future__ import annotations

import json
from typing import List, Dict, Any

from prompt.entities_mentions import SYSTEM_PROMPT_ENTS, USER_PROMPT_ENTS

from pydantic import ValidationError

from llm import LLMClient
from schema.entities_mentions import EntitiesOutput, EntityNoId


def _extract_json_str(s: str) -> str:
    """
    Try to extract the JSON object from a possibly noisy string:
    returns substring from first '{' to last '}'.
    """
    if not s:
        raise ValueError("Empty LLM output.")
    lb = s.find("{")
    rb = s.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        raise ValueError("No JSON object found in LLM output.")
    return s[lb : rb + 1]


def _validate_and_fix_spans(parsed: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
    """
    Enforce span bounds within the chunk and drop invalid mentions/entities.
    """
    max_idx = len(chunk_text)
    cleaned = {"entities": []}
    for ent in parsed.get("entities", []):
        kept_mentions = []
        for m in ent.get("mentions", []):
            start, end, sid = m.get("start"), m.get("end"), m.get("sent_id")
            if isinstance(start, int) and isinstance(end, int) and isinstance(sid, int):
                if 0 <= start < end <= max_idx:
                    kept_mentions.append({"start": start, "end": end, "sent_id": sid})
        if kept_mentions:
            cleaned["entities"].append(
                {
                    "type": ent.get("type"),
                    "name": ent.get("name"),
                    "mentions": kept_mentions,
                }
            )
    return cleaned


def _assign_ids(entities: List[EntityNoId], chunk_id: str) -> List[Dict[str, Any]]:
    """Give stable, deterministic IDs per chunk order."""
    out = []
    for i, ent in enumerate(entities):
        out.append(
            {
                "id": f"{chunk_id}_e{i}",
                "type": ent.type.value,
                "name": ent.name,
                "mentions": [m.dict() for m in ent.mentions],
            }
        )
    return out


def p1_entities_mentions(
    llm: LLMClient, chunk_id: str, chunk_text: str, max_retries: int = 2
) -> Dict[str, Any]:
    """
    Run P1: Entities & Mentions over a chunk using the provided LLM client.

    Args:
        llm: Reusable LLM client.
        chunk_id: Stable ID for this chunk.
        chunk_text: The text content for this chunk.
        max_retries: Number of schema-fix retries.

    Returns:
        {
          "chunk_id": str,
          "entities": [
             {"id": "...", "type": "Person|Org|Location|Artifact|Concept",
              "name": "...",
              "mentions":[{"start":int,"end":int,"sent_id":int}]}
          ]
        }
    """
    system = SYSTEM_PROMPT_ENTS
    user = USER_PROMPT_ENTS.substitute(chunk_id=chunk_id, chunk_text=chunk_text)

    last_error = None
    for attempt in range(max_retries + 1):
        raw = llm.generate(system, user)

        try:
            js = _extract_json_str(raw)
            parsed = json.loads(js)
        except Exception as e:
            last_error = e
            continue

        # Hard span sanitation
        parsed = _validate_and_fix_spans(parsed, chunk_text)

        # Pydantic validation
        try:
            valid = EntitiesOutput(**parsed)
        except ValidationError as e:
            last_error = e
            continue

        # Assign chunk-scoped IDs and return
        with_ids = _assign_ids(valid.entities, chunk_id)
        return {"chunk_id": chunk_id, "entities": with_ids}

    # If we get here, all attempts failed
    raise RuntimeError(f"P1 failed after {max_retries + 1} attempts: {last_error}")
