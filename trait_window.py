from __future__ import annotations

import json
from typing import Any, Dict, List
from pydantic import ValidationError

from prompt.trait_window import SYSTEM_PROMPT_ENTS, USER_PROMPT_ENTS
from schema.trait_window import Output


def p4_trait_window(
    llm, person_name: str, window_text: str, require_spans: bool = True
) -> Dict[str, Any]:
    user = USER_PROMPT_ENTS.substitute(person_name=person_name, window_text=window_text)
    raw = llm.generate(SYSTEM_PROMPT_ENTS, user)
    lb, rb = raw.find("{"), raw.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        return {
            "person": person_name,
            "trait_probs": {},
            "evidence": [],
            "rationale": None,
        }

    parsed = json.loads(raw[lb : rb + 1])
    try:
        out = Output.model_validate(parsed).model_dump()
    except ValidationError:
        return {
            "person": person_name,
            "trait_probs": {},
            "evidence": [],
            "rationale": None,
        }

    if require_spans and not out["evidence"]:
        # insist on spans for faithfulness
        out["trait_probs"] = {}
    return out


# ---- Calibration (stub; plug your fitted scaler) ----
def calibrate_trait_probs(trait_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Apply your learned isotonic/temperature scaling.
    For now, identity with clamping to [0,1].
    """
    return {k: max(0.0, min(1.0, float(v))) for k, v in (trait_probs or {}).items()}


# ---- Convert window JSON to standardized evidence items (ready for R3) ----
def convert_trait_json_to_evidence(
    person_id: str,
    window_meta: Dict[str, Any],  # has "start" (chunk offset) and "sent_ids"
    trait_probs: Dict[str, float],
    evidence: List[Dict[str, Any]],
    method_prior: float = 0.70,  # prior reliability for LLM method
    context_quality: float = 0.90,  # window proximity
) -> List[Dict[str, Any]]:
    items = []
    base_c = method_prior * context_quality
    for trait, s in (trait_probs or {}).items():
        # attach first matching evidence item (or none)
        ev_span = evidence[0]["span"] if evidence else [0, 0]
        start_w, end_w = ev_span if len(ev_span) == 2 else [0, 0]
        items.append(
            {
                "person": person_id,
                "trait": trait,
                "s": float(s),  # strength
                "c": float(base_c),  # local confidence
                "span": {
                    "chunk_id": window_meta.get("chunk_id"),
                    "start": window_meta["start"] + start_w,
                    "end": window_meta["start"] + end_w,
                    "sent_ids": window_meta.get("sent_ids", []),
                },
                "method": ["llm"],
                "rationale": None,
            }
        )
    return items
