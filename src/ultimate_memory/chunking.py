from __future__ import annotations

import re


def _find_fence_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char ranges for fenced code blocks (``` or ~~~)."""
    ranges: list[tuple[int, int]] = []
    fence_re = re.compile(r"^(?P<marker>`{3,}|~{3,})[^\n]*$", re.MULTILINE)
    matches = list(fence_re.finditer(text))
    i = 0
    while i < len(matches):
        open_match = matches[i]
        open_marker = open_match.group("marker")[0]  # ` or ~
        open_len = len(open_match.group("marker"))
        # Find the matching close fence (same char, same or greater run length)
        for j in range(i + 1, len(matches)):
            close_match = matches[j]
            close_char = close_match.group("marker")[0]
            close_len = len(close_match.group("marker"))
            if close_char == open_marker and close_len >= open_len:
                ranges.append((open_match.start(), close_match.end()))
                i = j + 1
                break
        else:
            # Unclosed fence — treat to end of text
            ranges.append((open_match.start(), len(text)))
            break
    return ranges


def _in_fence(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(s < pos < e for s, e in ranges)


def _find_split(window: str, fences: list[tuple[int, int]], global_start: int) -> int:
    """Return the best split offset within *window* (relative to window start).

    Priority: heading boundary > paragraph break > sentence end > newline.
    Never splits inside a code fence.
    """
    min_offset = max(int(len(window) * 0.4), 1)

    def ok(offset: int) -> bool:
        return offset >= min_offset and not _in_fence(global_start + offset, fences)

    # 1. Last heading boundary (split just before the heading so it opens next chunk)
    heading_hits = [m.start() for m in re.finditer(r"\n(?=#{1,6} )", window) if ok(m.start())]
    if heading_hits:
        return heading_hits[-1]

    # 2. Last paragraph break (two or more newlines)
    para_hits = [m.end() for m in re.finditer(r"\n{2,}", window) if ok(m.start())]
    if para_hits:
        return para_hits[-1]

    # 3. Last sentence end (". " or ".\n")
    sent_hits = [m.end() for m in re.finditer(r"\. |\.\n", window) if ok(m.start())]
    if sent_hits:
        return sent_hits[-1]

    # 4. Last newline
    nl = window.rfind("\n", min_offset)
    if nl != -1 and ok(nl):
        return nl + 1

    return len(window)


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 160) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]

    fences = _find_fence_ranges(normalized)
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))

        if end >= len(normalized):
            tail = normalized[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # If the cut point lands inside a fence, extend to the fence's closing line
        # (capped at 2× chunk_chars to avoid runaway giant chunks).
        for fs, fe in fences:
            if fs < end < fe:
                end = min(fe, start + chunk_chars * 2)
                break

        if end >= len(normalized):
            tail = normalized[start:].strip()
            if tail:
                chunks.append(tail)
            break

        window = normalized[start:end]
        split_offset = _find_split(window, fences, start)
        actual_end = start + split_offset

        # Guard: never stall
        if actual_end <= start:
            actual_end = end

        chunk = normalized[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(0, actual_end - overlap)
        # Snap the overlap start out of any open fence
        for fs, fe in fences:
            if fs <= start < fe:
                start = fe
                break

    return [c for c in chunks if c]
