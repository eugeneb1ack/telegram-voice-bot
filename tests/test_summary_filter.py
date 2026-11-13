import asyncio
from dataclasses import replace

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendRichMessage
from test_dialogue import setup, voice
from test_dialogue_summary import music_dialogue
from test_journal import idle, runtime, stop
from test_journal import voice as durable_voice
from test_rendering import ParsedReply
from test_service import identified_message, make_dependencies

from voice_bot.dialogue_summary import has_summary_content
from voice_bot.providers import SummaryContext, summary_length_hint


@pytest.mark.parametrize(
    "text",
    [
        "Ну, эээ… ммм.",
        " Привет!\nПривет. ",
        "Всем привет!",
        "Спасибо большое, пожалуйста.",
        "Добрый день. Ну… здравствуйте.",
        "…",
        "Спасибо большое. " * 5000,
    ],
)
def test_only_complete_small_talk_is_skipped(text):
    assert not has_summary_content(text)


@pytest.mark.parametrize(
    "text",
    [
        "Да.",
        "Нет.",
        "Согласен.",
        "Не согласен.",
        "Когда?",
        "Сколько?",
        "В 17:00.",
        "Спасибо за помощь с отчётом.",
        "Привет, завтра не приду.",
        "Ну, отправь файл.",
        "Возможно.",
        "Это привет от Маши.",
        "Hello!",
        "Спасибо большое. " * 5000 + "Нет.",
    ],
)
def test_short_facts_answers_and_unknown_phrases_remain_eligible(text):
    assert has_summary_content(text)


async def test_long_small_talk_keeps_exact_transcript_without_summary_request(settings):
    service, bot, _ = make_dependencies(settings)
    source = "  Ну, эээ…\n\nСпасибо большое!  \n"
    service.transcriber.transcribe.return_value = source
    message = identified_message(1, duration=40)
    await service.handle(message, bot)
    service.summarizer.summarize.assert_not_awaited()
    assert ParsedReply(message.answer.await_args.args[0]).transcript == source
    assert "Коротко" not in message.answer.await_args.args[0]
    assert service.processed.reply_ids(message)
    await service.close()


@pytest.mark.parametrize("first_block", [("Привет.", "Ну, эээ.", "Спасибо."), ("Привет.",)])
async def test_skipped_block_stays_closed_after_restart(settings, first_block):
    service, bot, _ = setup(settings)
    for number, source in enumerate(first_block, 1):
        service.transcriber.transcribe.return_value = source
        item = identified_message(number, duration=40 if len(first_block) == 1 else 10)
        await service.handle(item, bot)
        assert ParsedReply(item.answer.await_args.args[0]).transcript == source
        assert "Коротко" not in item.answer.await_args.args[0]
    service.summarizer.summarize.assert_not_awaited()
    await service.close()

    service, bot, _ = setup(settings)
    for number in (4, 5, 6):
        item = await voice(service, bot, number)
        assert ("Коротко" in item.answer.await_args.args[0]) == (number == 6)
    service.summarizer.summarize.assert_awaited_once()
    context = service.summarizer.summarize.await_args.kwargs["context"]
    assert [node.message_id for node in context.dialogue.messages] == [4, 5, 6]
    await service.close()


async def test_greeting_does_not_hide_useful_pending_voice(settings):
    service, bot, _ = setup(settings)
    await voice(service, bot, 1)
    service.transcriber.transcribe.return_value = "Спасибо."
    await service.handle(identified_message(2, duration=40, seconds=30), bot)
    service.summarizer.summarize.assert_awaited_once()
    source = service.summarizer.summarize.await_args.args[0]
    assert "Мысль 1." in source and "Спасибо." in source
    await service.close()


async def test_prepared_skipped_summary_survives_restart_without_model_calls(settings, tmp_path):
    path = tmp_path / "queue.db"
    sent = []
    service, bot, journal = runtime(settings, path, sent)
    transcript = "  Ну, эээ…\nСпасибо!  "
    service.transcriber.transcribe.return_value = transcript
    entered = asyncio.Event()

    async def before_delivery(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    service._deliver_prepared = before_delivery
    await service.accept(durable_voice().as_(bot), bot)
    await asyncio.wait_for(entered.wait(), 2)
    await service.queue.close(0)
    await stop(service, bot, journal)

    service, bot, journal = runtime(settings, path, sent)
    await service.recover(bot)
    await idle(service)
    assert len(sent) == 1
    assert "Коротко" not in sent[0].text
    assert ParsedReply(sent[0].text).transcript == transcript
    service.transcriber.transcribe.assert_not_awaited()
    service.summarizer.summarize.assert_not_awaited()
    assert journal.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    await stop(service, bot, journal)


@pytest.mark.parametrize("fallback", [False, True])
async def test_forwarded_small_talk_batch_has_transcript_without_fake_summary(settings, fallback):
    service, bot, _ = make_dependencies(settings)
    if fallback:
        bot.send_rich_message.side_effect = TelegramBadRequest(
            method=SendRichMessage(chat_id=123, rich_message={"blocks": []}),
            message="unsupported",
        )
    service.transcriber.transcribe.side_effect = ["Привет.", "Спасибо."]
    messages = [identified_message(i, forwarded=True) for i in (1, 2)]
    await service._handle_messages(messages, bot)
    service.summarizer.summarize.assert_not_awaited()
    expected = "Голосовое 1/2:\nПривет.\n\nГолосовое 2/2:\nСпасибо."
    if fallback:
        response = messages[-1].answer.await_args.args[0]
        assert "Коротко" not in response
        assert ParsedReply(response).transcript == expected
    else:
        rich = bot.send_rich_message.await_args.kwargs["rich_message"]
        assert "Коротко" not in rich.model_dump_json()
        assert rich.blocks[-1].blocks[0].blocks[0].text == expected
    await service.close()


def test_dialogue_detail_budget_counts_only_the_three_primary_voices(settings):
    graph = music_dialogue()
    graph = replace(
        graph,
        messages=tuple(replace(node, text="деталь " * 70) for node in graph.messages),
    )
    context = SummaryContext(dialogue=graph)
    assert "93–140" in summary_length_hint("service labels are not speech", context, settings)
    one = replace(graph, branches=(replace(graph.branches[0], message_ids=(10,)),))
    # Recipient and unselected nodes cannot inflate either word or voice count.
    assert "30–40" in summary_length_hint("ignored", replace(context, dialogue=one), settings)
