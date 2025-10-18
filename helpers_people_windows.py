from __future__ import annotations
from typing import Any, Dict, List


def persons_in_chunk(
    entities_in_chunk: List[Dict[str, Any]],
    coref_map_local_to_global: Dict[str, str],
    chunk_id: str,
) -> List[Dict[str, Any]]:
    people = []
    for e in entities_in_chunk:
        if e.get("type") != "Person":
            continue
        can_id = coref_map_local_to_global.get(f"{chunk_id}:{e['id']}")
        if not can_id:
            continue
        people.append(
            {
                "canonical_id": can_id,
                "display_name": e["name"],
                "local_ids": [e["id"]],
                "mentions": e.get("mentions", []),
            }
        )
    return people


def windows_for_person(
    chunk_text: str,
    sent_index: List[Dict[str, int]],  # [{"sent_id":0,"start":..,"end":..}, ...]
    mentions: List[Dict[str, int]],
    window_size: int = 1,
) -> List[Dict[str, Any]]:
    """Return sentence±window_size windows for a person's mentions."""
    sent_ids = sorted({m["sent_id"] for m in mentions})
    windows = []
    for sid in sent_ids:
        sids = list(
            range(
                max(0, sid - window_size), min(len(sent_index), sid + window_size + 1)
            )
        )
        start = min(sent_index[i]["start"] for i in sids)
        end = max(sent_index[i]["end"] for i in sids)
        windows.append(
            {
                "sent_id_center": sid,
                "sent_ids": sids,
                "start": start,
                "end": end,
                "text": chunk_text[start:end],
            }
        )
    return windows
