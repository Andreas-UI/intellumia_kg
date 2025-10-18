from string import Template

SYSTEM_PROMPT_ENTS = """
You are an information extraction engine.
Output STRICT JSON that conforms to the provided schema. Do not add commentary.
"""

USER_PROMPT_ENTS = Template(
    """
    CHUNK_ID: $chunk_id
    TEXT (the chunk):
    <<<
    $chunk_text
    >>>

    TASK:
    1) Extract entities that appear in the TEXT. Allowed types: Person, Org, Location, Artifact, Concept.
    2) For each entity, return a canonical 'name' and ALL mention spans as [start,end,sent_id], where indices are RELATIVE TO THIS CHUNK.
    3) Spans must be exact character offsets within the given TEXT.
    4) If nothing is found, return {"entities": []}.
    5) Return exactly one JSON object. No code fences, no explanations.
    6) Output STRICT JSON ONLY. Match this schema exactly:

    Schema:
    {"entities":[
    {"type":"Person|Org|Location|Artifact|Concept",
    "name":"string",
    "mentions":[{"start":int,"end":int,"sent_id":int}]}
    ]}
    """
)
