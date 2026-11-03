import re
from html.parser import HTMLParser

import pytest

from voice_bot.rendering import render_batch, render_replies, split_text, utf16_length


class ParsedReply(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.plain = ""
        self.transcript = ""
        self.in_blockquote = False
        self.inner_heading = False
        self.stack = []
        self.feed(text)
        assert self.stack == []
        if self.inner_heading:
            self.transcript = re.sub(
                r"^Расшифровка(?: \(\d+/\d+\))?:\n", "", self.transcript, count=1
            )

    def handle_starttag(self, tag, attrs):
        assert tag in {"b", "blockquote"}
        assert attrs == ([("expandable", None)] if tag == "blockquote" else [])
        if tag == "b" and self.in_blockquote:
            self.inner_heading = True
        self.stack.append(tag)
        if tag == "blockquote":
            self.in_blockquote = True

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag
        if tag == "blockquote":
            self.in_blockquote = False

    def handle_data(self, data):
        self.plain += data
        if self.in_blockquote:
            self.transcript += data


@pytest.mark.parametrize(
    "transcript",
    [
        "Привет <b>мир</b> & </blockquote><blockquote expandable> <script>alert(1)</script>",
        "😀<&\n" * 4000,
        "👨‍👩‍👧‍👦а́\n" * 1200,
        " без потерь пробелов \n" * 500,
    ],
)
@pytest.mark.parametrize("summary", [None, "Короткое <summary> &"])
def test_full_transcript_is_collapsible_escaped_and_lossless(transcript, summary):
    replies = [ParsedReply(reply) for reply in render_replies(summary, transcript)]
    assert "".join(reply.transcript for reply in replies) == transcript
    assert all(utf16_length(reply.plain) <= 4096 for reply in replies)


def test_short_note_is_one_reply():
    replies = render_replies("Встреча в пять.", "Встречаемся сегодня в пять.")
    assert len(replies) == 1
    assert ParsedReply(replies[0]).transcript == "Встречаемся сегодня в пять."


def test_transcript_only_reply_has_heading_inside_expandable_block():
    reply = render_replies(None, "\n  Точный текст\n")[0]
    assert reply.startswith("<blockquote expandable><b>Расшифровка:</b>\n")
    assert ParsedReply(reply).transcript == "\n  Точный текст\n"


def test_summary_name_is_escaped_without_separate_author_heading():
    reply = render_replies("<reader/> & test: встреча завтра.", "Встречаемся завтра.")[0]
    assert reply.startswith("<b>Коротко:</b>\n&lt;reader/&gt; &amp; test: встреча завтра.")
    assert "Автор:" not in reply
    assert "<blockquote expandable><b>Расшифровка:</b>\n" in reply
    assert ParsedReply(reply).transcript == "Встречаемся завтра."


def test_split_replies_have_no_separate_author_heading():
    replies = render_replies("мысль 😀" * 1000, "текст\n" * 2000)
    parsed = [ParsedReply(reply) for reply in replies]
    assert all(utf16_length(reply.plain) <= 4096 for reply in parsed)
    assert all("Автор:" not in reply.plain for reply in parsed)
    assert "".join(reply.transcript for reply in parsed) == "текст\n" * 2000


def test_long_summary_and_pathological_grapheme_do_not_break_limit():
    transcript = "а" + "\u0301" * 5000
    replies = render_replies("Большое summary 😀" * 1000, transcript)
    parsed = [ParsedReply(reply) for reply in replies]
    assert "".join(reply.transcript for reply in parsed) == transcript
    assert all(utf16_length(reply.plain) <= 4096 for reply in parsed)


def test_graphemes_are_not_split_and_input_is_preserved():
    text = "👩‍💻а́" * 20
    chunks = split_text(text, 12)
    assert "".join(chunks) == text
    assert all(not chunk.startswith(("\u200d", "\u0301")) for chunk in chunks)
    assert all(utf16_length(chunk) <= 12 for chunk in chunks)


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("summary", [None, "Первая тема.\n\nВторая тема."])
def test_batch_is_closed_details_with_plain_title_and_full_quoted_text(count, summary):
    transcript = "  <b>не форматирование</b>\n😀 & текст\n" * 1000
    rich = render_batch(summary, transcript, count)
    assert rich.skip_entity_detection is True
    if count > 1:
        assert rich.blocks[0].text.text == f"Получено голосовых: {count}"
    details = rich.blocks[-1]
    assert details.summary == "Расшифровка" and details.is_open is False
    assert details.blocks[0].type == "blockquote"
    assert details.blocks[0].blocks[0].text == transcript
    if summary:
        paragraphs = rich.blocks[:-1]
        assert [block.text for block in paragraphs if isinstance(block.text, str)] == [
            "Первая тема.",
            "Вторая тема.",
        ]


def test_three_voice_heading_shares_reply_with_full_current_transcript():
    transcript = "  Третья запись <&>\n\nс пробелами.  "
    replies = render_replies("Итог трёх.", transcript, summary_title="Коротко · 3 голосовых:")
    assert len(replies) == 1
    assert replies[0].startswith("<b>Коротко · 3 голосовых:</b>")
    assert ParsedReply(replies[0]).transcript == transcript


def test_three_voice_heading_preserves_parsed_limit_for_long_unicode_transcript():
    transcript = "👩‍💻 <&>\n" * 1000
    replies = render_replies("Итог трёх.", transcript, summary_title="Коротко · 3 голосовых:")
    parsed = [ParsedReply(reply) for reply in replies]
    assert all(utf16_length(reply.plain) <= 4096 for reply in parsed)
    assert "".join(reply.transcript for reply in parsed) == transcript
