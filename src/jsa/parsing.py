"""PDF text extraction, one record per physical page.

Page numbers are 1-based physical PDF pages, not printed page labels; the answer key
and citations use the same convention.
"""

import io
import logging
from typing import BinaryIO

from pypdf import PdfReader


def extract_document_pages(source_path: str, content: bytes) -> list[tuple[int, str]]:
    """Dispatch on extension: PDFs page by page, plain text split on form feeds."""
    if source_path.lower().endswith(".txt"):
        return [(n, text) for n, text in enumerate(content.decode("utf-8").split("\f"), start=1)]
    return extract_pages(content)


def extract_pages(pdf: bytes | BinaryIO) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` for every page; unreadable pages yield empty text."""
    stream = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    # pypdf logs a warning for every malformed object; blackline/scanned filings are full of them.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(stream)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not drop the document
            text = ""
        pages.append((number, text))
    return pages
