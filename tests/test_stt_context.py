"""STT receives bounded spelling hints without borrowing another conversation."""

import asyncio

import httpx
import pytest
from aiogram.types import User
from test_conversation_scope import scoped_voice
from test_journal import idle, runtime, stop, voice
from test_reply_context import foreign_message, reply_voice
from test_service import make_dependencies

from voice_bot.errors import ProviderError
from voice_bot.providers import APIClient, HTTPTranscriber
from voice_bot.reply_context import ReplyContext
from voice_bot.service import conversation_position
from voice_bot.stt_context import bounded_keywords, vocabulary


@pytest.fixture
async def stt_runtime(settings):
    service, bot, paths = make_dependencies(settings)
    try:
        yield service, bot, paths
    finally:
        await service.close()


def remember(service, message, text, *, key=None, number=1, seconds=-15, kind="voice"):
    service.memory.remember(
        key or conversation_position(message)[0],
        message.date.timestamp() + seconds,
        (number,),
        text,
        kind=kind,
    )


@pytest.mark.parametrize(
    "other_key", [(-200, 0, 222), (-100, 42, 222), (-100, 0, 333), (-100, 0, 222, 1)]
)
async def test_ambient_keywords_are_isolated_by_chat_topic_author_and_forward_scope(
    stt_runtime, other_key
):
    service, _, _ = stt_runtime
    message = scoped_voice(20)
    remember(service, message, "Qdrant")
    remember(service, message, "Forbidden", key=other_key)

    assert service._transcription_keywords(message) == ("Qdrant",)


@pytest.mark.parametrize("topic", [False, None, True])
async def test_ordinary_reply_thread_uses_zero_scope_and_real_topic_stays_separate(
    stt_runtime, topic
):
    service, _, _ = stt_runtime
    message = scoped_voice(20, thread=42, topic=topic)
    remember(service, message, "Ordinary", key=(-100, 0, 222))
    remember(service, message, "Forum", key=(-100, 42, 222))

    assert service._transcription_keywords(message) == (("Forum",) if topic else ("Ordinary",))


@pytest.mark.parametrize(
    "seconds,number,included",
    [
        (-181, 1, False),
        (-180, 1, True),
        (0, 19, True),
        (0, 20, False),
        (0, 21, False),
        (1, 1, False),
    ],
)
async def test_only_earlier_messages_within_series_gap_supply_keywords(
    stt_runtime, seconds, number, included
):
    service, _, _ = stt_runtime
    message = scoped_voice(20)
    remember(service, message, "Qdrant", number=number, seconds=seconds)

    assert service._transcription_keywords(message) == (("Qdrant",) if included else ())


async def test_only_last_three_eligible_sources_are_used_and_typed_spelling_has_priority(
    stt_runtime,
):
    service, _, _ = stt_runtime
    message = scoped_voice(20)
    for number, text, kind in [
        (1, "Obsolete", "text"),
        (2, "PostgreSQL", "text"),
        (3, "FastAPI", "voice"),
        (4, "Qdrant", "voice"),
    ]:
        remember(service, message, text, number=number, seconds=-100 + number * 10, kind=kind)

    assert service._transcription_keywords(message) == ("PostgreSQL", "Qdrant", "FastAPI")


async def test_typed_sources_precede_referenced_recognition_when_budget_is_full(stt_runtime):
    service, _, _ = stt_runtime
    target = foreign_message(kind="voice")
    message = reply_voice(20, target)
    message.caption = "CaptionTerm"
    recognized = tuple("Recognized" + chr(65 + index) for index in range(16))
    service.memory.remember((-100, 0, 444), target.date.timestamp(), (1,), " ".join(recognized))
    remember(service, message, "PostgreSQL", number=2, kind="text")

    assert service._transcription_keywords(message) == (
        "CaptionTerm",
        "PostgreSQL",
        *recognized[:14],
    )


@pytest.mark.parametrize("kind", ["text", "document"])
async def test_explicit_reply_text_or_attachment_caption_from_bot_is_allowed(stt_runtime, kind):
    service, bot, _ = stt_runtime
    target = foreign_message(kind=kind).model_copy(
        update={
            "text" if kind == "text" else "caption": "Qdrant",
            "from_user": User(id=444, is_bot=True, first_name="Helper"),
        }
    )

    assert service._transcription_keywords(reply_voice(target=target)) == ("Qdrant",)
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()


@pytest.mark.parametrize("restriction", ["other_chat", "receiver", "ephemeral"])
async def test_cross_chat_or_private_reply_text_cannot_supply_keywords(stt_runtime, restriction):
    service, bot, _ = stt_runtime
    target = foreign_message(chat=-200 if restriction == "other_chat" else -100).model_copy(
        update={"text": "Forbidden"}
    )
    if restriction == "receiver":
        target = target.model_copy(
            update={"receiver_user": User(id=222, is_bot=False, first_name="Reader")}
        )
    elif restriction == "ephemeral":
        target = target.model_copy(update={"ephemeral_message_id": 123})

    assert service._transcription_keywords(reply_voice(target=target)) == ()
    bot.get_file.assert_not_awaited()
    service.transcriber.transcribe.assert_not_awaited()


@pytest.mark.parametrize("source_kind", ["entry", "reply_context"])
async def test_referenced_voice_uses_exact_saved_text_without_transcribing_source_again(
    stt_runtime, source_kind
):
    service, bot, _ = stt_runtime
    target = foreign_message(kind="voice")
    message = reply_voice(target=target)
    service.memory.remember((-200, 0, 444), target.date.timestamp(), (1,), "Forbidden")
    if source_kind == "entry":
        service.memory.remember((-100, 0, 444), target.date.timestamp(), (1,), "Qdrant")
    else:
        service.memory.remember(
            (-100, 0, 555),
            target.date.timestamp(),
            (6,),
            "собственный текст",
            reply_contexts=(ReplyContext(6, 1, "Ирина", None, "voice", "Qdrant"),),
        )

    await service._transcribe(message, bot)

    service.transcriber.transcribe.assert_awaited_once()
    assert service.transcriber.transcribe.await_args.kwargs == {"keywords": ("Qdrant",)}
    bot.get_file.assert_awaited_once_with(message.voice.file_id)


@pytest.mark.parametrize("explicit_reply", [False, True])
async def test_private_text_delivery_does_not_read_ambient_public_context(
    stt_runtime, explicit_reply
):
    service, _, _ = stt_runtime
    target = foreign_message().model_copy(update={"text": "Qdrant"}) if explicit_reply else None
    message = reply_voice(target=target)
    remember(service, message, "Forbidden")
    message.receiver_user = User(id=222, is_bot=False, first_name="Reader")

    assert service._transcription_keywords(message) == (("Qdrant",) if explicit_reply else ())


def test_keyword_filter_keeps_literal_words_and_deduplicates_case_insensitively():
    assert bounded_keywords(
        [
            "Qdrant",
            "qdrant",
            "O’Connor",
            "Санкт-Петербург",
            "x",
            "12345",
            "<tag>",
            "ignore previous",
            "A" * 41,
            "\nSecret",
            "safe_word",
            "https://x",
            "@person",
            "Байкал",
        ]
    ) == ("Qdrant", "O’Connor", "Санкт-Петербург", "Байкал")


def test_keyword_count_is_bounded_to_sixteen():
    words = tuple("Term" + chr(65 + index) for index in range(26))
    assert bounded_keywords(words) == words[:16]


def test_product_spellings_keep_digits_and_hyphens_but_plain_numbers_are_not_hints():
    words = ("GPT-6", "AC-42", "term42", "12345")
    assert bounded_keywords(words) == words[:3]
    assert vocabulary([" ".join(words)]) == words[:2]


def test_total_keyword_characters_are_bounded_without_cutting_words():
    words = tuple(chr(65 + index) * 40 for index in range(12))
    assert bounded_keywords(words) == words[:8]
    assert sum(map(len, bounded_keywords(words))) == 320


def test_vocabulary_ignores_links_handles_lowercase_and_unbounded_source_tail():
    assert vocabulary(
        ["https://Secret.example/Forbidden @Hidden Qdrant qdrant lower PascalCase"]
    ) == ("Qdrant", "PascalCase")
    assert vocabulary(["a " * 500] * 3 + ["Forbidden"]) == ()


@pytest.mark.parametrize(
    "text", ["Росэлектро…", "Ири\u0301на", "a " * 495 + "Electroencephalography"]
)
def test_vocabulary_never_promotes_a_partial_spelling_to_a_keyword(text):
    assert vocabulary([text]) == ()


@pytest.mark.parametrize("model", ["gpt-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
async def test_multipart_keywords_preserve_model_language_and_exact_transcript(
    settings, tmp_path, model
):
    settings.stt_model = model
    settings.stt_language = "ru"
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"complete-audio")
    transcript = "  Первые слова.\n\nПоследние слова.  \n"

    def respond(request):
        assert request.url.path == "/v1/audio/transcriptions"
        body = request.content
        assert model.encode() in body and b"complete-audio" in body
        assert b'name="prompt"' not in body
        assert (b'name="keywords[]"' in body) is (model == "gpt-transcribe")
        assert (b'name="languages[]"' in body) is (model == "gpt-transcribe")
        assert (b'name="language"' in body) is (model != "gpt-transcribe")
        if model == "gpt-transcribe":
            assert body.count(b'name="keywords[]"') == 2
            assert b"\r\n\r\nQdrant\r\n" in body
            assert b"\r\n\r\nPostgreSQL\r\n" in body
        assert b"\r\n\r\nru\r\n" in body
        return httpx.Response(200, json={"text": transcript})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPTranscriber(APIClient(client, settings)).transcribe(
            audio, keywords=("Qdrant", "qdrant", "PostgreSQL", "ignore previous", "A" * 41)
        )
    assert result == transcript


async def test_retry_reuses_frozen_keywords_then_cached_transcript(stt_runtime):
    service, bot, paths = stt_runtime
    message = scoped_voice(20)
    remember(service, message, "Qdrant")
    transcript = "  Расшифровка.\n\nПродолжение.  "
    service.transcriber.transcribe.side_effect = [ProviderError("Нет ответа"), transcript]
    with pytest.raises(ProviderError):
        await service._transcribe(message, bot)
    remember(service, message, "NewTerm", number=2, seconds=-5)
    assert service._transcription_keywords(message) == ("NewTerm", "Qdrant")

    assert await service._transcribe(message, bot) == transcript
    assert await service._transcribe(message, bot) == transcript

    assert [call.kwargs for call in service.transcriber.transcribe.await_args_list] == [
        {"keywords": ("Qdrant",)},
        {"keywords": ("Qdrant",)},
    ]
    assert bot.get_file.await_count == 2
    assert len(paths) == 2 and all(not path.exists() for path in paths)


@pytest.mark.parametrize("initial_word", ["Qdrant", ""])
async def test_restart_reuses_frozen_keywords_even_if_context_changes(
    settings, tmp_path, initial_word
):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    message = voice(10)
    if initial_word:
        remember(service, message, initial_word)
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    service.transcriber.transcribe.side_effect = blocked
    await service.accept(message.as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)

    restarted, bot, journal = runtime(settings, path, sent)
    try:
        remember(restarted, message, "NewTerm", number=2, seconds=-5)
        assert "NewTerm" in restarted._transcription_keywords(message)
        await restarted.recover(bot)
        await idle(restarted)

        restarted.transcriber.transcribe.assert_awaited_once()
        assert restarted.transcriber.transcribe.await_args.kwargs == (
            {"keywords": (initial_word,)} if initial_word else {}
        )
        assert len(sent) == 1
    finally:
        await stop(restarted, bot, journal)


async def test_own_caption_survives_admission_restart_before_keywords_are_frozen(
    settings, tmp_path
):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    message = voice(10).model_copy(update={"caption": "Qdrant — название проекта."})
    job_id = journal.admit(message, None, collecting=False, charged=False)
    assert job_id is not None and journal.get(job_id)["cache"] == {}
    await stop(service, bot, journal)

    restarted, bot, journal = runtime(settings, path, sent)
    try:
        await restarted.recover(bot)
        await idle(restarted)

        restarted.transcriber.transcribe.assert_awaited_once()
        assert restarted.transcriber.transcribe.await_args.kwargs == {"keywords": ("Qdrant",)}
        assert len(sent) == 1
    finally:
        await stop(restarted, bot, journal)


async def test_restart_after_stt_reuses_whole_transcript_without_new_keywords_or_download(
    settings, tmp_path
):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    message = voice(10)
    remember(service, message, "Qdrant")
    transcript = "  Исходный текст.\n\nПродолжение.  "
    service.transcriber.transcribe.return_value = transcript
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    service.summarizer.summarize.side_effect = blocked
    await service.accept(message.as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)

    restarted, bot, journal = runtime(settings, path, sent)
    try:
        remember(restarted, message, "NewTerm", number=2, seconds=-5)
        await restarted.recover(bot)
        await idle(restarted)

        restarted.transcriber.transcribe.assert_not_awaited()
        bot.get_file.assert_not_awaited()
        restarted.summarizer.summarize.assert_awaited_once()
        assert restarted.summarizer.summarize.await_args.args[0] == transcript
        assert len(sent) == 1 and transcript in sent[0].text
    finally:
        await stop(restarted, bot, journal)
