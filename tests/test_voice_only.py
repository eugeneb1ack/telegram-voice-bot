import json
from dataclasses import asdict

from test_dialogue import setup, text, voice
from test_reply_context import foreign_message, process_voice

from voice_bot.reply_context import as_voice


async def test_text_between_voices_is_not_summary_material(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await text(service, 2, "UnrelatedCaption")
    await voice(service, bot, 3)
    await voice(service, bot, 4)
    call = service.summarizer.summarize.await_args
    assert "UnrelatedCaption" not in call.args[0]
    assert "UnrelatedCaption" not in json.dumps(asdict(call.kwargs["context"]))
    assert service.transcriber.transcribe.await_count == 3
    await service.close()


async def test_reply_keeps_recipient_without_loading_their_message(settings):
    service, bot, _ = setup(settings)
    target = foreign_message().model_copy(update={"text": "UnrelatedOldTopic"})
    await process_voice(service, bot, 10, target)
    await process_voice(service, bot, 11)
    await process_voice(service, bot, 12)
    context = service.summarizer.summarize.await_args.kwargs["context"]
    serialized = json.dumps(asdict(context), ensure_ascii=False)
    assert "UnrelatedOldTopic" not in serialized
    assert "Ирина" in serialized
    assert service.transcriber.transcribe.await_count == 3
    await service.close()


async def test_video_note_is_ignored_without_advancing_voice_counter(settings):
    service, bot, _ = setup(settings)
    note = foreign_message(chat=123, kind="video_note")
    await service.handle(note, bot)
    assert as_voice(note) is None
    service.transcriber.transcribe.assert_not_awaited()
    assert not service.memory.sessions
    for number in range(2, 5):
        await voice(service, bot, number, author=444)
        assert service.summarizer.summarize.await_count == int(number == 4)
    await service.close()


async def test_legacy_snapshot_cannot_restore_video_notes_as_voices(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    await voice(service, bot, 2)
    await service.close()
    path = settings.context_state_file
    snapshot = json.loads(path.read_text())
    snapshot["version"] = 1
    path.write_text(json.dumps(snapshot))
    service, bot, _ = setup(settings)
    assert service.memory.available and not service.memory.sessions
    for number in (3, 4, 5):
        await voice(service, bot, number)
        assert service.summarizer.summarize.await_count == int(number == 5)
    assert "Мысль 1" not in service.summarizer.summarize.await_args.args[0]
    await service.close()
    assert json.loads(path.read_text())["version"] == 2


async def test_failed_summary_cannot_supply_hints_to_next_block(settings):
    from test_service import identified_message

    from voice_bot.errors import ProviderError

    service, bot, _ = setup(settings)
    service.summarizer.summarize.side_effect = ProviderError("Unavailable")
    for number in (1, 2, 3):
        await voice(service, bot, number)
    assert service._transcription_keywords(identified_message(4, duration=10)) == ()
    await service.close()


async def test_recovered_video_note_is_skipped_and_voice_is_preserved(settings, tmp_path):
    from test_journal import idle, runtime, stop
    from test_journal import voice as queued_voice

    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    note = foreign_message(kind="video_note").model_copy(
        update={"voice": foreign_message(kind="voice").voice}
    )
    keys = [
        journal.admit(item, None, collecting=False, charged=False)
        for item in (note, queued_voice(2))
    ]
    job = journal.merge(keys)
    await service.recover(bot)
    await idle(service)
    assert len(sent) == 1
    service.transcriber.transcribe.assert_awaited_once()
    assert bot.get_file.await_args.args == ("file-2",)
    assert journal.seen(job)
    assert journal.recover(bot) == []
    await stop(service, bot, journal)


def test_pending_mixed_batch_filter_is_atomic_and_preserves_cached_voice(tmp_path):
    from test_journal import voice as queued_voice

    from voice_bot.journal import QueueJournal

    journal = QueueJournal(tmp_path / "queue.db")
    note = foreign_message(kind="video_note").model_copy(
        update={"voice": foreign_message(kind="voice").voice}
    )
    keys = [
        journal.admit(item, None, collecting=False, charged=False)
        for item in (note, queued_voice(2))
    ]
    job = journal.merge(keys)
    journal.set(job, "cache", {"saved-voice": "Exact transcript"})
    journal.set(job, "result", {"summary": "Old mixed summary"})
    assert journal.retain_voices(job)
    row = journal.get(job)
    assert len(row["payload"]) == 1 and row["payload"][0]["message"]["message_id"] == 2
    assert row["result"] is None and row["cache"]["saved-voice"] == "Exact transcript"
    journal.set(job, "result", {"summary": "New voice-only summary"})
    assert journal.retain_voices(job)
    assert journal.get(job)["result"] == {"summary": "New voice-only summary"}
    journal.close()
