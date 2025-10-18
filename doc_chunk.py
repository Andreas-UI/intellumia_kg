from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import os
import re


def build_sent_index(chunk_text: str) -> List[Dict]:
    """
    Lightweight sentence splitter that preserves offsets (relative to chunk).
    Splits on ., !, ?, and newlines while keeping abbreviations reasonably safe.
    """
    spans = []
    i, n = 0, len(chunk_text)
    sent_start = 0
    # Regex finds sentence boundaries: punctuation + space/newline OR two newlines
    boundary = re.compile(r'([.!?]+["”\']?\s+|\n{2,})')
    for m in boundary.finditer(chunk_text):
        end = m.end()
        # Trim trailing whitespace from the sentence span
        s = chunk_text[sent_start:end]
        # Find last non-space
        r = len(s) - 1
        while r >= 0 and s[r].isspace():
            r -= 1
        if r >= 0:
            spans.append(
                {"sent_id": len(spans), "start": sent_start, "end": sent_start + r + 1}
            )
        sent_start = end
    # Tail
    if sent_start < n:
        tail = chunk_text[sent_start:n]
        r = len(tail) - 1
        while r >= 0 and tail[r].isspace():
            r -= 1
        if r >= 0:
            spans.append(
                {
                    "sent_id": len(spans),
                    "start": sent_start,
                    "end": n if r == len(tail) - 1 else sent_start + r + 1,
                }
            )
    return spans


def doc_chunk(
    path_file: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 750,
    encoding: str = "utf-8",
) -> List[Dict]:
    """
    Chunk text and attach a sentence index to each chunk.
    """
    with open(path_file, "r", encoding=encoding) as f:
        text = f.read()  # DO NOT lowercase; keep raw for offsets

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)

    basename = os.path.splitext(os.path.basename(path_file))[0]
    chunked_docs: List[Dict] = []

    # Compute offsets incrementally (no .find needed)
    start_idx = 0
    for i, chunk in enumerate(chunks):
        end_idx = start_idx + len(chunk)
        chunked_docs.append(
            {
                "chunk_id": f"{basename}_{i}",
                "text": chunk,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "sent_index": build_sent_index(chunk),  # <-- add sentence index
            }
        )
        # next chunk begins at end - overlap
        start_idx = max(0, end_idx - chunk_overlap)

    return chunked_docs
