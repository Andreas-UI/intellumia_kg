from string import Template

SYSTEM_PROMPT_ENTS = """
You extract Big Five personality evidence from a short window of text.
Return STRICT JSON only.
"""

USER_PROMPT_ENTS = Template(
    """PERSON: $person_name
    WINDOW:
    <<<
    $window_text
    >>>

    TASK:
    - If the window clearly supports durable personality dispositions for PERSON, return trait probabilities in [0,1]
    for any of: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.
    - Cite EXACT evidence spans relative to THIS WINDOW using [start,end] (no sent_id needed here).
    - Include hedges (e.g., "sometimes") and negation (true/false) for each evidence item.
    - If uncertain, omit the trait(s). STRICT JSON only.

    JSON:
    {
    "person":"$person_name",
    "trait_probs": {"Conscientiousness": 0.84},
    "evidence": [
        {"span":[start,end], "text":"...", "hedges":[], "neg": false}
    ],
    "rationale":"one sentence"
    }
    """
)
