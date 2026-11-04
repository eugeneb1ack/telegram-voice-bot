"""A bounded shared conversation, with explicit authors and reply edges."""

from dataclasses import dataclass

from voice_bot.conversation import ConversationKey, ConversationMemory, Entry
from voice_bot.reply_context import ReplyContext


@dataclass(frozen=True)
class DialogueMessage:
    message_id: int
    author_id: int | None
    author_name: str | None
    author_username: str | None
    kind: str
    text: str
    reply_to_message_id: int | None = None
    unavailable_reason: str | None = None
    sent_at: float | None = None


@dataclass(frozen=True)
class DialogueBranch:
    branch_id: str
    recipient_id: int | None
    recipient_name: str | None
    recipient_username: str | None
    message_ids: tuple[int, ...]
    is_reply: bool = False


@dataclass(frozen=True)
class SharedDialogue:
    focus_author_id: int
    messages: tuple[DialogueMessage, ...]
    branches: tuple[DialogueBranch, ...]
    partial: bool = False
    has_previous_summary: bool = False


def shared_dialogue(
    memory: ConversationMemory,
    key: ConversationKey,
    sources: list[Entry],
    references: tuple[ReplyContext, ...],
    sent_at: float,
    message_id: int,
    *,
    partial: bool = False,
) -> SharedDialogue:
    """Current voice block and direct recipient identities; never conversation history."""
    anchor = memory.sessions[key].anchor
    refs = {ref.message_id: ref for ref in references}
    nodes: dict[int, DialogueMessage] = {}
    groups: dict[int | None, tuple[ReplyContext | None, list[int]]] = {}
    for entry in sources:
        mid = max(entry.message_ids)
        if entry.kind != "voice" or entry.sent_at > sent_at or mid > message_id:
            continue
        ref = refs.get(mid)
        if ref is not None and ref.reply_to_message_id >= mid:
            ref = None
        nodes[mid] = DialogueMessage(
            mid,
            key[2],
            anchor.author_name if anchor else None,
            anchor.author_username if anchor else None,
            "voice",
            entry.text,
            ref.reply_to_message_id if ref else None,
            sent_at=entry.sent_at,
        )
        recipient = ref.author_id if ref else None
        if recipient not in groups:
            groups[recipient] = (ref, [])
        groups[recipient][1].append(mid)
    for mid, node in tuple(nodes.items()):
        if node.reply_to_message_id is not None:
            ref = refs[mid]
            nodes.setdefault(
                ref.reply_to_message_id,
                DialogueMessage(
                    ref.reply_to_message_id,
                    ref.author_id,
                    ref.author_name,
                    ref.author_username,
                    "recipient",
                    "",
                ),
            )
    branches = tuple(
        DialogueBranch(
            f"b{index}",
            ref.author_id if ref else None,
            ref.author_name if ref else None,
            ref.author_username if ref else None,
            tuple(ids),
            ref is not None,
        )
        for index, (ref, ids) in enumerate(groups.values(), 1)
    )
    return SharedDialogue(key[2], tuple(nodes.values()), branches, partial=partial)
