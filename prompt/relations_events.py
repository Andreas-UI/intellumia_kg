from string import Template

SYSTEM_PROMPT_ENTS = """
You extract relations and events with exact spans.
Return STRICT JSON only; no commentary.
"""

USER_PROMPT_ENTS = Template(
    """
    CHUNK_ID: $chunk_id
    ENTITIES (local_id:name:type):
    $entity_table

    TEXT:
    <<<
    $chunk_text
    >>>

    TASK:
    1) Extract RELATIONS (STRICT JSON):
    {
        "relations":[
        {
            "type":"<short snake_case label derived from the exact surface phrase>",  // e.g., "works_at", "founded"
            "source_label":"<exact surface phrase from TEXT>",                        // REQUIRED, from text
            "head":"<local_id>",
            "tail":"<local_id>",
            "polarity":"asserted|negated|hypothetical",
            "modality":"factual|reported|conditional",
            "supports":[{"start":int,"end":int,"sent_id":int,"trigger":{"start":int,"end":int}}]
        }
        ]
    }

    2) Extract EVENTS (STRICT JSON):
    {
        "events":[
        {
            "type":"<short snake_case event label>",   // e.g., "award_ceremony", "meeting"
            "source_label":"<exact surface phrase from TEXT>", // REQUIRED
            "participants":[{"role":"...","entity_id":"<local_id>"}],
            "status":"planned|ongoing|completed|cancelled",
            "polarity":"asserted|negated|hypothetical",
            "supports":[{"start":int,"end":int,"sent_id":int,"trigger":{"start":int,"end":int}}]
        }
        ]
    }

    Rules:
    - Spans are RELATIVE to this CHUNK.
    - Each item MUST have ≥1 support span and a trigger INSIDE that support.
    - Use ONLY local entity IDs listed above.
    - If you cannot infer a good relation/event label from the text, set "type":"involved_in".
    - Output STRICT JSON ONLY: {"relations":[...], "events":[...]}.
    """
)
