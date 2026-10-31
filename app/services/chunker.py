import re
from collections import Counter

CHUNK_WORDS = 160
OVERLAP_WORDS = 25


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    repeated_lines: set[str] = set()
    if len(pages) >= 4:
        line_counts: Counter[str] = Counter()
        for _, text in pages:
            line_counts.update(
                {re.sub(r"\s+", " ", line).strip().casefold() for line in text.splitlines() if line.strip() and len(line.strip()) < 80}
            )
        repeated_lines = {
            line for line, count in line_counts.items() if count > len(pages) / 2
        }

    cleaned_pages = []
    for page_number, text in pages:
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        paragraphs: list[str] = []
        current_lines: list[str] = []
        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            key = normalized.casefold()
            if normalized and key in repeated_lines and len(normalized) < 80:
                continue
            if not normalized:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
            else:
                current_lines.append(normalized)
        if current_lines:
            paragraphs.append(" ".join(current_lines))
        cleaned_pages.append((page_number, "\n\n".join(paragraphs)))
    return cleaned_pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[tuple[int, str]]:
    chunks = []
    step = chunk_words - overlap_words
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            words_in_chunk = words[start : start + chunk_words]
            if words_in_chunk:
                chunks.append((page_number, " ".join(words_in_chunk)))
            if start + chunk_words >= len(words):
                break
    return chunks
