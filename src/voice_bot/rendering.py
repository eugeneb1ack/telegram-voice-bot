from html import escape

import regex
from aiogram.types import (
    InputRichBlockBlockQuotation,
    InputRichBlockDetails,
    InputRichBlockParagraph,
    InputRichMessage,
    RichTextBold,
)

MESSAGE_LIMIT = 4096
# Leave room for headings; count UTF-16 units and split before escaping.
CONTENT_LIMIT = 3500


def render_batch(summary: str | None, transcript: str, count: int) -> InputRichMessage:
    # Rich blocks preserve the transcript without HTML escaping.
    blocks = []
    if count > 1:
        blocks.append(
            InputRichBlockParagraph(text=RichTextBold(text=f"Получено голосовых: {count}"))
        )
    if summary is not None:
        blocks.append(InputRichBlockParagraph(text=RichTextBold(text="Коротко:")))
        blocks.extend(InputRichBlockParagraph(text=part) for part in summary.split("\n\n") if part)
    return InputRichMessage(
        skip_entity_detection=True,
        blocks=[
            *blocks,
            InputRichBlockDetails(
                summary="Расшифровка",
                is_open=False,
                blocks=[
                    InputRichBlockBlockQuotation(blocks=[InputRichBlockParagraph(text=transcript)])
                ],
            ),
        ],
    )


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_text(text: str, limit: int = CONTENT_LIMIT) -> list[str]:
    """Lossless split: keep graphemes together, including ZWJ emoji and combining marks.

    A pathological grapheme longer than the entire budget is split at code points;
    no code point, surrogate pair, whitespace or text is dropped.
    """
    if limit < 2:
        raise ValueError("limit must be at least two UTF-16 units")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for grapheme in regex.findall(r"\X", text):
        units = utf16_length(grapheme)
        pieces = list(grapheme) if units > limit else [grapheme]
        for piece in pieces:
            units = utf16_length(piece)
            if size + units > limit:
                chunks.append("".join(current))
                current, size = [], 0
            current.append(piece)
            size += units
    if current:
        chunks.append("".join(current))
    return chunks


def render_replies(
    summary: str | None, transcript: str, *, summary_title: str = "Коротко:"
) -> list[str]:
    transcript_parts = split_text(transcript)
    if summary is None:
        return [
            f"<blockquote expandable><b>Расшифровка:</b>\n{escape(part)}</blockquote>"
            for part in transcript_parts
        ]
    summary_parts = split_text(summary)
    # Usual voice note: one reply containing summary + collapsible full transcription.
    combined_plain = f"{summary_title}\n{summary}\n\nРасшифровка:\n{transcript}"
    if utf16_length(combined_plain) <= MESSAGE_LIMIT:
        return [
            f"<b>{escape(summary_title)}</b>\n{escape(summary)}\n\n"
            f"<blockquote expandable><b>Расшифровка:</b>\n{escape(transcript)}</blockquote>"
        ]
    replies = [f"<b>{escape(summary_title)}</b>\n{escape(part)}" for part in summary_parts]
    for index, part in enumerate(transcript_parts, 1):
        replies.append(
            f"<blockquote expandable><b>Расшифровка ({index}/{len(transcript_parts)}):</b>\n"
            f"{escape(part)}</blockquote>"
        )
    return replies
