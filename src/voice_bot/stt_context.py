"""Small literal vocabulary hints, never instructions or a replacement transcript."""

import re
import unicodedata
from collections.abc import Iterable

_WORD = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)
_LINK = re.compile(r"https?://\S+|@[\w]+", re.IGNORECASE)
MAX_KEYWORDS = 16
MAX_KEYWORD_CHARS = 320


def bounded_keywords(words: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    budget = MAX_KEYWORD_CHARS
    for word in words:
        # Single literal words only: no markup, instructions, newlines or partial tokens.
        if (
            not 2 <= len(word) <= 40
            or _WORD.fullmatch(word) is None
            or not any(character.isalpha() for character in word)
        ):
            continue
        if word.casefold() in seen or len(word) > budget:
            continue
        result.append(word)
        seen.add(word.casefold())
        budget -= len(word)
        if len(result) == MAX_KEYWORDS:
            break
    return tuple(result)


def vocabulary(texts: Iterable[str]) -> tuple[str, ...]:
    # Capitalization is only a conservative spelling candidate, not entity recognition.
    words: list[str] = []
    budget = 3000
    for text in texts:
        if budget <= 0:
            break
        text = unicodedata.normalize("NFC", text)
        excerpt = text[: min(1000, budget)]
        budget -= len(excerpt)
        if len(excerpt) < len(text):
            excerpt = re.sub(r"\S+$", "", excerpt)
        excerpt = _LINK.sub("", excerpt)
        for match in _WORD.finditer(excerpt):
            token = match.group()
            before = excerpt[match.start() - 1 : match.start()] if match.start() else ""
            after = excerpt[match.end() : match.end() + 1]
            if (
                after == "…"
                or excerpt[match.end() :].startswith("...")
                or any(unicodedata.category(char).startswith("M") for char in before + after)
            ):
                continue
            if any(character.isupper() for character in token):
                words.append(token)
    return bounded_keywords(words)
