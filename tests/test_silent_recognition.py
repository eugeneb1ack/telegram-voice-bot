from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_dialogue import setup, voice
from test_journal import idle, runtime, stop
from test_journal import voice as durable_voice
from test_service import identified_message, make_dependencies

from voice_bot.errors import NoSpeechError, ProviderError


@pytest.mark.parametrize("result", ["", " \n\t", NoSpeechError(), ProviderError("SECRET")])
async def test_unrecognized_voice_sends_nothing_and_releases_resources(settings, result, caplog):
    service, bot, paths = make_dependencies(settings)
    if isinstance(result, Exception):
        service.transcriber.transcribe.side_effect = result
    else:
        service.transcriber.transcribe.return_value = result
    message = identified_message(1, duration=40)
    await service.handle(message, bot)
    message.answer.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert not service.processed.pending and service.active_jobs == 0
    assert not any(path.exists() for path in paths)
    assert "SECRET" not in caplog.text
    assert not any(s.entries for s in service.memory.sessions.values())
    await service.close()


async def test_silence_between_voices_does_not_advance_triplet(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    service.transcriber.transcribe.side_effect = NoSpeechError()
    silent = await voice(service, bot, 2)
    silent.answer.assert_not_awaited()
    service.transcriber.transcribe.side_effect = None
    third = await voice(service, bot, 3)
    third.answer.assert_awaited_once()
    assert "Коротко" not in third.answer.await_args.args[0]
    fourth = await voice(service, bot, 4)
    assert "Коротко · 3 голосовых:" in fourth.answer.await_args.args[0]
    sources = service.summarizer.summarize.await_args.args[0]
    assert all(f"Мысль {i}." in sources for i in (1, 3, 4))
    assert "Мысль 2." not in sources
    await service.close()


async def test_initial_silence_does_not_start_voice_window(settings):
    service, bot, _ = setup(settings)
    service.transcriber.transcribe.side_effect = NoSpeechError()
    await voice(service, bot, 1, seconds=0)
    service.transcriber.transcribe.side_effect = None
    for mid, seconds in ((2, 800), (3, 1000), (4, 1100)):
        last = await voice(service, bot, mid, seconds=seconds)
    assert "Коротко · 3 голосовых:" in last.answer.await_args.args[0]
    await service.close()


@pytest.mark.parametrize("all_silent", [False, True])
async def test_forwarded_batch_omits_silent_parts_without_error_text(settings, all_silent):
    service, bot, _ = make_dependencies(settings)
    bot.send_rich_message = AsyncMock(return_value=SimpleNamespace(message_id=1000))
    service.transcriber.transcribe.side_effect = [
        NoSpeechError(),
        NoSpeechError() if all_silent else "Полная речь",
        NoSpeechError(),
    ]
    messages = [identified_message(i, forwarded=True) for i in (1, 2, 3)]
    await service._handle_messages(messages, bot)
    assert all(message.answer.await_count == 0 for message in messages)
    if all_silent:
        bot.send_rich_message.assert_not_awaited()
        service.summarizer.summarize.assert_not_awaited()
    else:
        rich = bot.send_rich_message.await_args.kwargs["rich_message"]
        transcript = rich.blocks[-1].blocks[0].blocks[0].text
        assert transcript == "Голосовое 2/3:\nПолная речь"
        assert "Не удалось" not in rich.model_dump_json()
    assert service.processed.reply_ids(messages[0]) == ()
    assert service.processed.reply_ids(messages[2]) == ()
    if not all_silent:
        assert service.processed.reply_ids(messages[1])
    await service.close()


@pytest.mark.parametrize("crash_before_receipt", [False, True])
async def test_silent_result_survives_restart_without_reply_or_second_stt(
    settings, tmp_path, crash_before_receipt
):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    service.transcriber.transcribe.side_effect = NoSpeechError()
    if crash_before_receipt:

        async def fail(*args, **kwargs):
            raise OSError("receipt unavailable")

        service.processed.complete = fail
    await service.accept(durable_voice().as_(bot), bot)
    await idle(service)
    assert sent == []
    await stop(service, bot, journal)
    service, bot, journal = runtime(settings, path, sent)
    await service.recover(bot)
    await idle(service)
    await service.accept(durable_voice().as_(bot), bot)
    await idle(service)
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert sent == []
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    await stop(service, bot, journal)
