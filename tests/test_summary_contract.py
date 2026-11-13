import json

import httpx
import pytest
from aiogram.types import MessageOriginUser, User
from summary_helpers import completed_summary
from test_service import identified_message, make_dependencies

from voice_bot.dialogue_summary import SUMMARY_TEXT_FORMAT, SummaryOutput, parse_summary_response
from voice_bot.errors import NoSpeechError, ProviderError
from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext


@pytest.mark.parametrize(
    "case",
    [
        "missing_status",
        "missing_message_status",
        "missing_role",
        "wrong_role",
        "incomplete_message",
        "refusal",
        "error",
        "multiple_messages",
        "incomplete_details",
    ],
)
async def test_inconsistent_responses_never_publish_text_or_retry(settings, case):
    body = completed_summary()
    if case == "missing_status":
        del body["status"]
    elif case == "missing_message_status":
        del body["output"][0]["status"]
    elif case == "missing_role":
        del body["output"][0]["role"]
    elif case == "wrong_role":
        body["output"][0]["role"] = "user"
    elif case == "incomplete_message":
        body["output"][0]["status"] = "incomplete"
    elif case == "refusal":
        body["output"][0]["content"].insert(0, {"type": "refusal", "refusal": "SECRET"})
    elif case == "error":
        body["error"] = {"code": "server_error", "message": "SECRET"}
    elif case == "multiple_messages":
        body["output"].append(body["output"][0].copy())
    else:
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError) as exc:
            await HTTPSummarizer(APIClient(client, settings)).summarize("Завтра встреча в семь.")
    assert "SECRET" not in str(exc.value)
    assert len(calls) == 1


async def test_refusal_in_token_limited_response_is_not_retried(settings):
    body = completed_summary()
    body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    body["output"][0]["content"].insert(0, {"type": "refusal", "refusal": "SECRET"})
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError):
            await HTTPSummarizer(APIClient(client, settings)).summarize("Завтра встреча в семь.")
    assert len(calls) == 1


async def test_single_source_uses_strict_contract_and_sends_source_once(settings):
    source = "  Завтра встреча в семь.\n"

    def respond(request):
        payload = json.loads(request.content)
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
        data = json.loads(payload["input"])
        primary = data["primary_messages"]
        assert len(primary) == 1
        assert primary[0]["text"] == source
        assert primary[0]["speaker"]["name"] == "Анна"
        assert payload["input"].count("Завтра встреча в семь.") == 1
        return httpx.Response(200, json=completed_summary())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            source, context=SummaryContext(author_name="Анна")
        )
    assert result == "Анна: Завтра встреча в семь."


async def test_partial_batch_only_sends_recognized_speaker_and_preserves_source_labels(settings):
    service, bot, _ = make_dependencies(settings)
    source = "  Завтра встреча в семь.\n"
    service.transcriber.transcribe.side_effect = [NoSpeechError(), source]
    messages = [identified_message(i, forwarded=True) for i in (1, 2)]
    for item, name, uid in zip(messages, ("Silent", "Spoken"), (10, 20), strict=True):
        item.forward_origin = MessageOriginUser(
            date=item.date, sender_user=User(id=uid, is_bot=False, first_name=name)
        )

    def respond(request):
        payload = json.loads(request.content)
        data = json.loads(payload["input"])
        assert data["primary_messages"] == [
            {
                "message_id": 2,
                "speaker": {"id": 20, "name": "Spoken"},
                "text": source,
                "reply_recipient": None,
            }
        ]
        assert "Silent" not in payload["input"]
        assert payload["input"].count("Завтра встреча в семь.") == 1
        return httpx.Response(200, json=completed_summary())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        service.summarizer = HTTPSummarizer(APIClient(client, settings))
        await service._handle_messages(messages, bot)
    rich = bot.send_rich_message.await_args.kwargs["rich_message"]
    assert rich.blocks[2].text == "Spoken: Завтра встреча в семь."
    assert rich.blocks[-1].blocks[0].blocks[0].text == "Голосовое 2/2:\n" + source
    await service.close()


@pytest.mark.parametrize(
    "value", [None, {}, [], {"summary": 4}, {"summary": " \n"}, {"summary": "x", "extra": 1}]
)
def test_plain_and_dialogue_share_strict_output_schema(value):
    body = completed_summary()
    body["output"][0]["content"][0]["text"] = json.dumps(value)
    with pytest.raises(ProviderError):
        parse_summary_response(body)
    assert SUMMARY_TEXT_FORMAT["schema"] == SummaryOutput.model_json_schema()
    assert SUMMARY_TEXT_FORMAT["schema"]["additionalProperties"] is False


def test_reasoning_item_is_allowed_but_never_published():
    body = completed_summary("  Встреча в семь.\n")
    body["output"].insert(0, {"type": "reasoning", "id": "rs_test", "summary": []})
    assert parse_summary_response(body) == "Встреча в семь."


async def test_dialogue_validates_only_selected_sources(settings):
    from test_dialogue_summary import music_dialogue

    from voice_bot.providers import VoicePartContext

    def respond(request):
        primary = json.loads(json.loads(request.content)["input"])["primary_messages"]
        assert [source["message_id"] for source in primary] == [10, 11, 12]
        return httpx.Response(200, json=completed_summary("Антон любит джаз."))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await HTTPSummarizer(APIClient(client, settings)).summarize(
            "Неиспользуемая плоская расшифровка.",
            context=SummaryContext(
                author_name="Антон",
                dialogue=music_dialogue(),
                voices=(VoicePartContext(0, "Посторонний", None, None),),
            ),
        )
    assert result == "Антон любит джаз."


@pytest.mark.parametrize(
    "invalid", ["missing_text", "empty_text", "duplicate_id", "negative_id", "oversized"]
)
async def test_invalid_canonical_sources_fail_before_api(settings, invalid):
    from unittest.mock import AsyncMock

    from voice_bot.providers import VoicePartContext

    parts = [VoicePartContext(1, "Антон", None, 10, message_id=10, text="Встреча в семь.")]
    if invalid == "missing_text":
        parts = [VoicePartContext(1, "Антон", None, 10)]
    elif invalid == "empty_text":
        parts = [VoicePartContext(1, "Антон", None, 10, text=" \n")]
    elif invalid == "duplicate_id":
        parts *= 2
    elif invalid == "negative_id":
        parts = [VoicePartContext(1, "Антон", None, 10, message_id=-10, text="Встреча в семь.")]
    else:
        parts = [
            VoicePartContext(
                1, "Антон", None, 10, text="x" * (settings.summary_max_input_chars + 1)
            )
        ]
    api = APIClient(None, settings)
    api.post = AsyncMock()
    with pytest.raises(ProviderError):
        await HTTPSummarizer(api).summarize(
            "Короткая обёртка", context=SummaryContext(voices=tuple(parts))
        )
    api.post.assert_not_awaited()
