from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from prompt.relations_events import SYSTEM_PROMPT_ENTS, USER_PROMPT_ENTS
from schema.relations_events import Event, Output, Relation


# ---- Helpers ----
def _slugify(label: str, fallback: str = "involved_in") -> str:
    if not label or not label.strip():
        return fallback
    s = label.strip().lower()
    # collapse non-alphanum to underscore, trim repeats
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def _entity_table(entities_in_chunk: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{e['id']} : {e['name']} : {e['type']}" for e in entities_in_chunk
    )


def _extract_json(s: str) -> Dict[str, Any]:
    lb, rb = s.find("{"), s.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        return {"relations": [], "events": []}
    return json.loads(s[lb : rb + 1])


def _map_local_to_global(
    local_id: str, chunk_id: str, coref_map: Dict[str, str]
) -> Optional[str]:
    return coref_map.get(f"{chunk_id}:{local_id}")


def _clip_spans(obj: Dict[str, Any], text_len: int) -> bool:
    ok = True
    for sp in obj.get("supports", []):
        s, e = max(0, int(sp["start"])), min(text_len, int(sp["end"]))
        if s >= e:
            ok = False
        sp["start"], sp["end"] = s, e
        trig = sp.get("trigger")
        if trig:
            ts, te = int(trig.get("start", -1)), int(trig.get("end", -1))
            if not (s <= ts < te <= e):
                ok = False
    return ok


def p3_relations_events(
    llm,
    chunk_id: str,
    chunk_text: str,
    entities_in_chunk: List[Dict[str, Any]],
    coref_map_local_to_global: Dict[str, str],
) -> Dict[str, Any]:
    user = USER_PROMPT_ENTS.substitute(
        chunk_id=chunk_id,
        entity_table=_entity_table(entities_in_chunk),
        chunk_text=chunk_text,
    )
    raw = llm.generate(SYSTEM_PROMPT_ENTS, user)
    parsed = _extract_json(raw)

    text_len = len(chunk_text)
    out_rels, out_evts = [], []
    rel_idx, evt_idx = 0, 0

    # Relations
    for r in parsed.get("relations", []) or []:
        head = _map_local_to_global(
            r.get("head", ""), chunk_id, coref_map_local_to_global
        )
        tail = _map_local_to_global(
            r.get("tail", ""), chunk_id, coref_map_local_to_global
        )
        if not head or not tail:
            continue
        # type: derive from source_label if missing
        surface = (r.get("source_label") or "").strip()
        if not surface:
            # require a source phrase; if none, skip to keep quality
            continue
        norm_type = _slugify(r.get("type", "") or surface, fallback="involved_in")
        item = {
            "id": f"{chunk_id}_rel_{rel_idx}",
            "type": norm_type,
            "source_label": surface,
            "head": head,
            "tail": tail,
            "polarity": r.get("polarity", "asserted"),
            "modality": r.get("modality", "factual"),
            "supports": r.get("supports", []),
            "place": r.get("place"),
            "confidence_local": float(r.get("confidence_local", 0.7)),
            "rationale": r.get("rationale"),
        }
        if not _clip_spans(item, text_len):
            continue
        try:
            out_rels.append(Relation.model_validate(item).model_dump())
            rel_idx += 1
        except ValidationError:
            continue

    # Events
    for e in parsed.get("events", []) or []:
        surface = (e.get("source_label") or "").strip()
        if not surface:
            continue
        norm_type = _slugify(e.get("type", "") or surface, fallback="involved_in")
        item = {
            "id": f"{chunk_id}_evt_{evt_idx}",
            "type": norm_type,
            "source_label": surface,
            "participants": e.get("participants", []),
            "status": e.get("status", "completed"),
            "polarity": e.get("polarity", "asserted"),
            "supports": e.get("supports", []),
            "place": e.get("place"),
            "confidence_local": float(e.get("confidence_local", 0.7)),
            "rationale": e.get("rationale"),
        }
        if not _clip_spans(item, text_len):
            continue
        try:
            out_evts.append(Event.model_validate(item).model_dump())
            evt_idx += 1
        except ValidationError:
            continue

    return Output(chunk_id=chunk_id, relations=out_rels, events=out_evts).model_dump()
