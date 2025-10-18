from __future__ import annotations
from typing import Any, Dict, List, Tuple
from collections import defaultdict


def _rel_key(r: Dict[str, Any]) -> Tuple:
    return (r["type"], r["head"], r["tail"], (r.get("time") or {}).get("norm"))


def _evt_key(e: Dict[str, Any]) -> Tuple:
    parts = tuple(
        sorted((p["role"], p["entity_id"]) for p in e.get("participants", []))
    )
    return (e["type"], parts, (e.get("time") or {}).get("norm"))


def r2_fuse_relations_events(rels_ev_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Flatten
    rels, evts = [], []
    for ch in rels_ev_all:
        rels.extend(ch.get("relations", []))
        evts.extend(ch.get("events", []))

    # Fuse relations
    rel_bucket = defaultdict(list)
    for r in rels:
        rel_bucket[_rel_key(r)].append(r)

    fused_rels = []
    for key, group in rel_bucket.items():
        base = dict(group[0])
        # union supports; keep highest confidence
        supports = []
        conf = 0.0
        for g in group:
            supports.extend(g.get("supports", []))
            conf = max(conf, float(g.get("confidence_local", 0.0)))
        base["supports"] = supports
        base["confidence_local"] = conf
        fused_rels.append(base)

    # Fuse events
    evt_bucket = defaultdict(list)
    for e in evts:
        evt_bucket[_evt_key(e)].append(e)

    fused_evts = []
    for key, group in evt_bucket.items():
        base = dict(group[0])
        supports = []
        conf = 0.0
        for g in group:
            supports.extend(g.get("supports", []))
            conf = max(conf, float(g.get("confidence_local", 0.0)))
        base["supports"] = supports
        base["confidence_local"] = conf
        fused_evts.append(base)

    return {"relations": fused_rels, "events": fused_evts}
