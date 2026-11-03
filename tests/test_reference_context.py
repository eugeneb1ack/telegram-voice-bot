from dataclasses import replace

import pytest
from test_semantic_context import link

from voice_bot.conversation import Anchor, ConversationMemory, Entry
from voice_bot.dialogue_graph import leading_person, shared_dialogue
from voice_bot.dialogue_summary import dialogue_payload
from voice_bot.semantic_context import apply_links


def story(*, source_time=926, first_time=1000, last_time=1327, second_person=False):
    mem = ConversationMemory(
        420,
        12000,
        200,
        lambda: last_time,
        max_entries=12,
        series_gap=420,
        history_lookback=300,
    )
    key = (-100, 0, 1)
    mem.remember(key, source_time, (1,), "Вот Марк сейчас снимает такие ролики.", kind="text")
    if second_person:
        mem.remember(key, source_time + 1, (2,), "Олег тоже снимает такие ролики.", kind="text")
    sources = [
        Entry((10,), first_time, "Он недавно хвалил работы фотографа. Комментарии были смешными."),
        Entry((11,), first_time + 260, "Я из Казани."),
        Entry((12,), last_time, "Проверяю, сможет ли бот правильно назвать участников."),
    ]
    for entry in sources:
        mem.remember(
            key,
            entry.sent_at,
            entry.message_ids,
            entry.text,
            anchor=Anchor(max(entry.message_ids), "Антон", None, "supergroup"),
        )
    return mem, shared_dialogue(mem, key, sources, (), last_time, 12)


def person_link(**overrides):
    return link(
        relation="person_reference",
        current_quote="Он недавно хвалил работы фотографа",
        source_quote="Вот Марк сейчас снимает такие ролики",
        mention="Он",
        referent="Марк",
        **overrides,
    )


def test_person_reference_is_bound_to_its_voice_and_passed_as_grounded_background():
    _, graph = story()
    selected = apply_links(graph, {"links": [person_link()]})
    payload = dialogue_payload(selected, 10000)
    first, second, third = payload["branches"][0]["primary_messages"]
    assert first["references"] == ({"source_id": 1, "mention": "Он", "referent": "Марк"},)
    assert not second["references"] and not third["references"]
    assert first["text"].startswith("Он недавно")  # no rewriting the transcript
    assert payload["branches"][0]["background_messages"][0]["role"] == "same_speaker"


@pytest.mark.parametrize("age,allowed", [(299, True), (300, True), (301, False)])
def test_five_minutes_are_checked_before_each_voice_not_before_summary(age, allowed):
    _, graph = story(source_time=1000 - age, last_time=1419)
    assert bool(graph.candidates) is allowed
    selected = apply_links(graph, {"links": [person_link()]})
    assert bool(next(n for n in selected.messages if n.message_id == 10).references) is allowed


def test_source_allowed_for_first_voice_cannot_explain_a_later_voice_outside_window():
    _, graph = story(source_time=800)
    late = replace(graph.messages[-1], text=graph.messages[0].text)
    graph = replace(graph, messages=(*graph.messages[:-1], late))
    selected = apply_links(graph, {"links": [person_link(message_id=12)]})
    assert selected.branches[0].context_message_ids == ()


@pytest.mark.parametrize(
    "change",
    [
        {"referent": "Иван"},
        {"mention": "Она"},
        {"confidence": "ambiguous"},
    ],
)
def test_unproven_person_mapping_is_rejected(change):
    _, graph = story()
    evidence = person_link()
    evidence.update(change)
    selected = apply_links(graph, {"links": [evidence]})
    assert selected.branches[0].context_message_ids == ()
    assert not any(n.references for n in selected.messages)


def test_conflicting_names_for_one_mention_admit_neither_source():
    _, graph = story(second_person=True)
    second = person_link(source_id=2)
    second.update(source_quote="Олег тоже снимает такие ролики", referent="Олег")
    selected = apply_links(graph, {"links": [person_link(), second]})
    assert selected.branches[0].context_message_ids == ()
    assert not any(n.references for n in selected.messages)


def test_part_of_a_name_is_not_an_exact_person_reference():
    _, graph = story()
    graph = replace(
        graph, candidates=(replace(graph.candidates[0], text="Вот Марков сейчас снимает ролики."),)
    )
    evidence = person_link()
    evidence["source_quote"] = "Вот Марков сейчас снимает ролики"
    selected = apply_links(graph, {"links": [evidence]})
    assert selected.branches[0].context_message_ids == ()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Он недавно хвалил фотографии", "Он"),
        ("Ну, кстати, он тоже придёт", "он"),
        ("Я встретил нового соседа. Он придёт", None),
        ("Сосед собирает фотокнигу", None),
        ("Онлайн прошла встреча", None),
    ],
)
def test_only_leading_person_can_inherit_an_old_name(text, expected):
    assert leading_person(text) == expected


def test_unknown_opening_person_is_never_marked_as_the_speaker():
    _, graph = story()
    primary = dialogue_payload(graph, 10000)["branches"][0]["primary_messages"]
    assert primary[0]["unresolved_reference"] == "Он"
    assert not primary[1]["unresolved_reference"]


@pytest.mark.parametrize("direct_reply", [True, False])
def test_known_object_reference_is_not_marked_as_an_unknown_person(direct_reply):
    _, graph = story()
    source = replace(graph.candidates[0], text="Купил новый велосипед.")
    first = replace(
        graph.messages[0],
        text="Он лёгкий и удобный.",
        reply_to_message_id=source.message_id if direct_reply else None,
    )
    graph = replace(
        graph,
        messages=(source, first, *graph.messages[1:]),
        branches=tuple(
            replace(b, context_message_ids=(source.message_id,)) for b in graph.branches
        ),
    )
    branch = dialogue_payload(graph, 10000)["branches"][0]
    assert branch["primary_messages"][0]["unresolved_reference"] is None
    assert branch["background_messages"][0]["text"] == "Купил новый велосипед."


def test_person_in_a_later_new_topic_cannot_inherit_the_old_name():
    _, graph = story(last_time=1100)
    messages = (
        replace(graph.messages[0], text="Вчера встретил нового соседа."),
        replace(graph.messages[1], text="Он хочет сделать фотокнигу.", sent_at=1020),
        graph.messages[2],
    )
    graph = replace(graph, messages=messages)
    evidence = person_link(message_id=11)
    evidence["current_quote"] = "Он хочет сделать фотокнигу"
    selected = apply_links(graph, {"links": [evidence]})
    assert selected.branches[0].context_message_ids == ()


async def test_background_survives_restart_until_series_finishes_without_extending_counter(
    tmp_path,
):
    now = [0.0]
    path = tmp_path / "context.json"

    def memory():
        return ConversationMemory(
            420,
            12000,
            200,
            lambda: now[0],
            max_entries=12,
            path=path,
            series_gap=420,
            history_lookback=300,
        )

    mem = memory()
    key = (-100, 0, 1)
    mem.remember(key, 0, (1,), "Вот Марк снимает ролики", kind="text")
    now[0] = 300
    mem.remember(key, 300, (10,), "Он хвалил фотографа")
    now[0] = 650
    mem.remember(key, 650, (11,), "Уже не пишет комментариев")
    assert await mem.checkpoint()
    mem.clear()
    restored = memory()
    assert restored.available and restored.source(-100, 1) is not None
    assert restored.activity(key, 719).started_at == 300
    assert restored.activity(key, 720).started_at == 720
    assert restored.activity(key, 720).voice_count == 0
    restored.clear()
