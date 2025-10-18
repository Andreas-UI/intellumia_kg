from __future__ import annotations
from typing import Any, Dict, List
from collections import defaultdict


def r3_aggregate_traits_beta(
    evidence_items: List[Dict[str, Any]],
    tau_score: float = 0.65,
    tau_conf: float = 0.60,
) -> List[Dict[str, Any]]:
    # Group by (person, trait)
    bucket = defaultdict(list)
    for ev in evidence_items:
        bucket[(ev["person"], ev["trait"])].append(ev)

    edges = []
    for (person, trait), E in bucket.items():
        S = sum(e["s"] * e["c"] for e in E)
        N = sum(e["c"] for e in E)
        alpha0, beta0 = 1.0, 1.0
        alpha = alpha0 + S
        beta = beta0 + max(0.0, N - S)
        score = alpha / (alpha + beta)
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
        conf = max(0.0, 1.0 - 4.0 * var)

        label = (
            "HAS_TRAIT"
            if (score >= tau_score and conf >= tau_conf)
            else "CANDIDATE_TRAIT"
        )
        # top-k: pick top 3 by s*c
        topk = sorted(E, key=lambda x: x["s"] * x["c"], reverse=True)[:3]
        edges.append(
            {
                "person": person,
                "trait": trait,
                "score": score,
                "confidence": conf,
                "label": label,
                "n_evidence": len(E),
                "topk": [
                    {
                        "chunk_id": t["span"]["chunk_id"],
                        "start": t["span"]["start"],
                        "end": t["span"]["end"],
                    }
                    for t in topk
                ],
            }
        )
    return edges
