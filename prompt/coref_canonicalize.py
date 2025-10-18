from string import Template

SYSTEM_PROMPT_ENTS = """
You are a careful coreference adjudicator.
You decide if two entity mentions refer to the SAME real-world entity.
Return STRICT JSON only.
"""


USER_PROMPT_ENTS = Template(
    """
    TASK:
    Decide whether MENTION A and MENTION B refer to the SAME real-world entity.
    Consider the type, names, aliases, and the provided contexts.
    Be conservative: if you are uncertain, answer false.

    A:
    type: $type_a
    name: "$name_a"
    aliases: $aliases_a
    context: $context_a

    B:
    type: $type_b
    name: "$name_b"
    aliases: $aliases_b
    context: $context_b

    Return STRICT JSON:
    {"same": true|false, "confidence": 0..1, "rationale": "one short sentence"}
    """
)
