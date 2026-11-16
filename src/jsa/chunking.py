"""Chunking: fixed-size character windows per page."""

from dataclasses import dataclass

CHUNK_SIZE = 1000


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


def naive_chunks(doc_id: str, pages: list[tuple[int, str]], size: int = CHUNK_SIZE) -> list[Chunk]:
    """Split each page's text into consecutive ``size``-character windows, no overlap."""
    chunks = []
    for page, text in pages:
        text = text.strip()
        for i, start in enumerate(range(0, len(text), size)):
            chunks.append(Chunk(f"{doc_id}:{page}:{i}", doc_id, page, text[start : start + size]))
    return chunks
