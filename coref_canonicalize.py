from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from llm import LLMClient
from prompt.coref_canonicalize import SYSTEM_PROMPT_ENTS, USER_PROMPT_ENTS
from schema.coref_canonicalize import (
    CanonicalEntity,
    CorefDecision,
    CorefResult,
    P1ChunkEntities,
    P1Entity,
)
from schema.entities_mentions import Mention


def normalize_name(s: str) -> str:
    """Casefold + strip accents + collapse spaces and punctuation for blocking."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set_jaccard(a: str, b: str) -> float:
    A = set(normalize_name(a).split())
    B = set(normalize_name(b).split())
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def longest_common_subseq_len(a: str, b: str) -> int:
    # Lightweight LCS length (O(n*m)) for short names
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def name_similarity(a: str, b: str) -> float:
    """Blend Jaccard on tokens with normalized LCS to reduce LLM calls."""
    a_n = normalize_name(a)
    b_n = normalize_name(b)
    if not a_n or not b_n:
        return 0.0
    j = token_set_jaccard(a_n, b_n)
    lcs = longest_common_subseq_len(a_n, b_n) / max(1, max(len(a_n), len(b_n)))
    return 0.6 * j + 0.4 * lcs


def extract_context_snippet(
    chunk_texts: Optional[Dict],
    chunk_id: str,
    mentions: List[Mention],
    window_chars: int = 80,
) -> str:
    """Build a compact context snippet around the first mention span, if available."""
    if not chunk_texts or chunk_id not in chunk_texts or not mentions:
        return "null"
    text = chunk_texts[chunk_id]
    m = mentions[0]
    s = max(0, m.start - window_chars)
    e = min(len(text), m.end + window_chars)
    snippet = text[s:e].replace("\n", " ").strip()
    return json.dumps(snippet)  # keep quotes safe inside template


class DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def p2_coref_canonicalize(
    llm: LLMClient,
    p1_chunks: List[Dict[str, Any]],
    chunk_texts: Optional[Dict] = None,
    high_sim_threshold: float = 0.93,
    low_sim_threshold: float = 0.40,
    max_llm_retries: int = 3,
) -> Dict[str, Any]:
    """
    Coreference & canonicalization across P1 chunk entities.

    Args:
        llm: Reusable LLM client (from P1).
        p1_chunks: List of P1 outputs: [{"chunk_id": "...", "entities": [ {...}, ... ]}, ...]
        chunk_texts: Optional map chunk_id -> chunk raw text to build mention context.
        high_sim_threshold: >= threshold => deterministic merge (no LLM).
        low_sim_threshold: <= threshold => deterministic NOT same (no LLM).
        max_llm_retries: LLM retry budget for schema/parse issues.

    Returns:
        Dict compatible with CorefResult:
        {
          "entities_canonical": [
             {"id":"ent_0","type":"Person","name":"Rina","aliases":["Rina","Ms. Tan"],"members":["D1_0_e0","D1_3_e2"]},
             ...
          ],
          "coref_map": {
             "D1_0_e0": "ent_0",
             "D1_3_e2": "ent_0",
             ...
          }
        }
    """
    # Validate & coerce inputs to Pydantic v2 models
    chunk_models: List[P1ChunkEntities] = []
    for item in p1_chunks:
        chunk_models.append(P1ChunkEntities.model_validate(item))

    # Flatten mentions with global mention IDs
    mentions: Dict[str, P1Entity] = {}
    by_type: Dict[str, List[str]] = {}  # type -> list of mention_ids
    for ch in chunk_models:
        for ent in ch.entities:
            mid = f"{ch.chunk_id}:{ent.id}"
            mentions[mid] = ent
            by_type.setdefault(ent.type, []).append(mid)

    # Prepare DSU clusters
    dsu = DSU()

    # Fast deterministic merges by exact normalized name within same type
    for t, mids in by_type.items():
        # if t not in restrict_types:
        #     continue
        buckets: Dict[str, List[str]] = {}
        for mid in mids:
            key = normalize_name(mentions[mid].name)
            buckets.setdefault(key, []).append(mid)
        for group in buckets.values():
            if len(group) > 1:
                root = group[0]
                for g in group[1:]:
                    dsu.union(root, g)

    # Candidate pairs for adjudication across buckets
    def candidate_pairs(mids: List[str]) -> List[Tuple[str, str]]:
        pairs = []
        # Simple banded comparison by first letter to prune
        index: Dict[str, List[str]] = {}
        for mid in mids:
            nn = normalize_name(mentions[mid].name)
            if not nn:
                continue
            key = nn[:1]  # first character
            index.setdefault(key, []).append(mid)
        for band_mids in index.values():
            n = len(band_mids)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = band_mids[i], band_mids[j]
                    # skip if already merged
                    if dsu.find(a) == dsu.find(b):
                        continue
                    pairs.append((a, b))
        return pairs

    # Heuristic + LLM adjudication
    for t, mids in by_type.items():
        # if t not in restrict_types:
        #     continue
        for a, b in candidate_pairs(mids):
            na, nb = mentions[a].name, mentions[b].name
            sim = name_similarity(na, nb)

            if sim >= high_sim_threshold:
                dsu.union(a, b)
                continue
            if sim <= low_sim_threshold:
                continue  # definitely not

            # Ambiguous: ask the LLM
            A = mentions[a]
            B = mentions[b]
            aliases_a = list({A.name})
            aliases_b = list({B.name})

            ctx_a = (
                extract_context_snippet(chunk_texts, a.split(":")[0], A.mentions)
                if A.mentions
                else "null"
            )
            ctx_b = (
                extract_context_snippet(chunk_texts, b.split(":")[0], B.mentions)
                if B.mentions
                else "null"
            )

            user_prompt = USER_PROMPT_ENTS.substitute(
                type_a=A.type,
                name_a=A.name,
                aliases_a=json.dumps(aliases_a),
                context_a=ctx_a,
                type_b=B.type,
                name_b=B.name,
                aliases_b=json.dumps(aliases_b),
                context_b=ctx_b,
            )

            # Retry loop for LLM
            decision: Optional[CorefDecision] = None
            last_err: Optional[Exception] = None
            for _ in range(max_llm_retries + 1):
                raw = llm.generate(SYSTEM_PROMPT_ENTS, user_prompt)
                try:
                    # Extract JSON object from output
                    lb, rb = raw.find("{"), raw.rfind("}")
                    if lb == -1 or rb == -1 or rb <= lb:
                        raise ValueError("No JSON object found in LLM output.")
                    js = raw[lb : rb + 1]
                    parsed = json.loads(js)
                    decision = CorefDecision.model_validate(parsed)
                    break
                except Exception as e:
                    last_err = e
                    continue

            if decision and decision.same and decision.confidence >= 0.6:
                dsu.union(a, b)
            # else: leave them separate

    # Build clusters from DSU
    clusters: Dict[str, List[str]] = {}
    for mid in mentions.keys():
        rid = dsu.find(mid)
        clusters.setdefault(rid, []).append(mid)

    # Build canonical entities (display name = most frequent or longest name)
    canonical_entities: List[CanonicalEntity] = []
    coref_map: Dict[str, str] = {}

    def choose_canonical_name(ids: List[str]) -> str:
        names = [mentions[mid].name for mid in ids]
        # frequency, then longest
        freq: Dict[str, int] = {}
        for n in names:
            freq[n] = freq.get(n, 0) + 1
        best = sorted(names, key=lambda x: (freq[x], len(x)), reverse=True)[0]
        return best

    # Stable canonical IDs
    for idx, (root, mids) in enumerate(sorted(clusters.items(), key=lambda kv: kv[0])):
        types = {mentions[mid].type for mid in mids}
        # If mixed types slip in, pick the majority
        t = max(
            types, key=lambda ty: sum(1 for mid in mids if mentions[mid].type == ty)
        )
        can_id = f"ent_{idx}"
        can_name = choose_canonical_name(mids)
        alias_set = list(sorted(set(mentions[mid].name for mid in mids)))

        canonical = CanonicalEntity(
            id=can_id, type=t, name=can_name, aliases=alias_set, members=mids
        )
        canonical_entities.append(canonical)
        for mid in mids:
            coref_map[mid] = can_id

    result = CorefResult(entities_canonical=canonical_entities, coref_map=coref_map)
    return result.model_dump()
