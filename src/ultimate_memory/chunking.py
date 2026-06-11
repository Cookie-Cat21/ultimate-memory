from __future__ import annotations


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 160) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        window = normalized[start:end]
        if end < len(normalized):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if split_at > chunk_chars * 0.45:
                end = start + split_at + 1
                window = normalized[start:end]
        chunks.append(window.strip())
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]

