"""Безопасная нормализация текста документов. Без пересказа и правок нормы."""

from __future__ import annotations

import re
import unicodedata

SOFT_HYPHEN = "\u00ad"
NBSP = "\u00a0"
NARROW_NBSP = "\u202f"

HEADING_LINE = re.compile(
    r"(?im)^\s*(статья\s+\d+(?:\.\d+)*|глава\s+\d+|раздел\s+[ivxлc\d]+|"
    r"пункт\s+\d+(?:\.\d+)*|часть\s+\d+|приложение\s+[№n]?\s*\w+|"
    r"§\s*\d+)\b"
)
NUMBERED_ITEM = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]|[a-zа-яё]\)|[–—\-]\s|\*\s)", re.I)
LEGAL_KEEP = re.compile(
    r"(не\s+\w+|статья\s+\d+(?:\.\d+)*|\d{1,2}[./]\d{1,2}[./]\d{2,4}|"
    r"\d[\d\s]*[.,]?\d*\s*(?:руб|₽|%|процент))",
    re.I,
)


def normalize_unicode(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    if not text:
        return "", actions
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        actions.append("unicode_nfc")
    cleaned = nfc.replace(SOFT_HYPHEN, "").replace(NBSP, " ").replace(NARROW_NBSP, " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if cleaned != nfc:
        actions.append("technical_whitespace")
    return cleaned, actions


def restore_hyphenation(text: str) -> tuple[str, list[str]]:
    restored, n = re.subn(r"(?<=\w)-\n(?=\w)", "", text)
    if n:
        return restored, ["restore_hyphenation"]
    return text, []


def collapse_technical_spaces(text: str) -> tuple[str, list[str]]:
    collapsed = re.sub(r"[ \t]+\n", "\n", text)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
    if collapsed != text:
        return collapsed.strip("\n"), ["collapse_spaces"]
    return collapsed.strip("\n"), []


def strip_repeated_headers(page_texts: list[str]) -> tuple[list[str], list[str]]:
    """Удаляет колонтитул только если одна и та же строка на ≥3 страницах."""
    if len(page_texts) < 3:
        return page_texts, []
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for page in page_texts:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
        last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1
    drop: set[str] = set()
    for line, count in first_lines.items():
        if count >= 3 and len(line) <= 80 and not LEGAL_KEEP.search(line):
            drop.add(line)
    for line, count in last_lines.items():
        if count >= 3 and len(line) <= 80 and not LEGAL_KEEP.search(line):
            drop.add(line)
    if not drop:
        return page_texts, []
    cleaned = []
    for page in page_texts:
        kept = [ln for ln in page.splitlines() if ln.strip() not in drop]
        cleaned.append("\n".join(kept))
    return cleaned, ["strip_repeated_headers"]


def normalize_element_text(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    current, acts = normalize_unicode(text or "")
    actions.extend(acts)
    current, acts = restore_hyphenation(current)
    actions.extend(acts)
    current, acts = collapse_technical_spaces(current)
    actions.extend(acts)
    return current, actions


def looks_like_heading(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 180:
        return False
    if HEADING_LINE.match(stripped):
        return True
    if stripped.isupper() and any(ch.isalpha() for ch in stripped) and 3 <= len(stripped) <= 80:
        return True
    return False


def looks_like_list_item(line: str) -> bool:
    return bool(NUMBERED_ITEM.match(line or ""))


def classify_legal_block(text: str) -> str | None:
    stripped = (text or "").strip()
    if re.match(r"(?i)^статья\s+\d+", stripped):
        return "article"
    if re.match(r"(?i)^(пункт|часть|пп\.|п\.)\s*\d+", stripped):
        return "clause"
    if re.match(r"(?i)^приложение\b", stripped):
        return "appendix"
    if re.match(r"(?i)^сноска\b", stripped):
        return "footnote"
    return None
