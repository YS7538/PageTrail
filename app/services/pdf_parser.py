from pathlib import Path

import fitz


def extract_pages(path: Path) -> list[tuple[int, str]]:
    try:
        with fitz.open(path) as pdf:
            return [(page_number, page.get_text()) for page_number, page in enumerate(pdf, start=1)]
    except Exception as exc:
        raise ValueError("Could not open file as a PDF") from exc
