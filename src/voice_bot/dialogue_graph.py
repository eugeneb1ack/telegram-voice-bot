"""A bounded shared conversation, with explicit authors and reply edges."""

from dataclasses import dataclass

from voice_bot.conversation import ConversationKey, ConversationMemory, Entry
from voice_bot.reply_context import ReplyContext

MAX_REPLY_DEPTH = 3


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


@dataclass(frozen=True)
class DialogueBranch:
    branch_id: str
    recipient_id: int | None
    recipient_name: str | None
    recipient_username: str | None
    message_ids: tuple[int, ...]
    is_reply: bool = False
    context_message_ids: tuple[int, ...] = ()


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
    """Select only focus messages and up to three explicit reply ancestors."""
    nodes: dict[int, DialogueMessage] = {}
    own = memory.sessions[key]
    anchor = own.anchor
    for entry in sources:
        mid = max(entry.message_ids)
        ref = next((r for r in references if r.message_id == mid), None)
        nodes[mid] = DialogueMessage(
            mid,
            key[2],
            anchor.author_name if anchor else None,
            anchor.author_username if anchor else None,
            entry.kind,
            entry.text,
            ref.reply_to_message_id if ref else None,
        )
    focus_ids = tuple(nodes)
    # The full focus text has its existing source budget. Shared background has one
    # further chat-wide budget, not one extra budget for each of 10–20 participants.
    budget = memory.max_chars
    candidates: dict[int, DialogueMessage] = {}
    edges: dict[int, ReplyContext] = {}
    for other, session in memory.sessions.items():
        if len(other) != 3 or other[:2] != key[:2]:
            continue
        author = session.anchor
        for entry in session.entries:
            mid = max(entry.message_ids)
            if len(entry.message_ids) != 1 or mid in nodes:
                continue
            if not (0 <= sent_at - entry.sent_at <= memory.ttl and mid < message_id):
                continue
            ref = next((r for r in entry.reply_contexts if r.message_id == mid), None)
            if ref is not None:
                edges[mid] = ref
            # A recap may cover unrelated topics/recipients. An exact reply must
            # resolve to this recording's transcript, never its aggregate recap.
            candidates[mid] = DialogueMessage(
                mid,
                other[2],
                author.author_name if author else None,
                author.author_username if author else None,
                entry.kind,
                entry.text,
                ref.reply_to_message_id if ref else None,
            )
    edges.update((ref.message_id, ref) for ref in references if ref.message_id in nodes)
    # Breadth first: all direct targets precede more distant ancestors. A cached
    # target in another forum topic is allowed only as an explicit leaf; we do not
    # crawl that topic's history. Siblings and replies TO our sources never qualify.
    frontier = list(focus_ids)
    for _ in range(MAX_REPLY_DEPTH):
        following: list[int] = []
        for mid in frontier:
            ref = edges.get(mid)
            if ref is None or ref.reply_to_message_id >= mid:
                continue
            target_id = ref.reply_to_message_id
            if target_id in nodes:
                following.append(target_id)
                continue
            if len(nodes) >= 100:
                continue
            cached = candidates.get(target_id)
            text = ref.text or (cached.text if cached else "")
            limit = min(budget, 3000)
            if len(text) > limit:
                text = text[: max(0, limit - 1)] + ("…" if limit else "")
            budget -= len(text)
            nodes[target_id] = DialogueMessage(
                target_id,
                cached.author_id if cached else ref.author_id,
                cached.author_name if cached else ref.author_name,
                cached.author_username if cached else ref.author_username,
                ref.kind,
                text,
                cached.reply_to_message_id if cached else None,
                (ref.unavailable_reason or "Контекст недоступен или ограничен лимитом")
                if not text
                else None,
            )
            following.append(target_id)
        frontier = following

    groups: dict[tuple[str, int], tuple[DialogueMessage | None, list[int]]] = {}
    for mid in focus_ids:
        target = nodes.get(nodes[mid].reply_to_message_id or 0)
        seen = {mid}
        # An explicit reply to one's own earlier message continues that reply chain.
        while target is not None and target.author_id == key[2] and target.message_id not in seen:
            seen.add(target.message_id)
            target = nodes.get(target.reply_to_message_id or 0)
        if target is not None and target.message_id not in seen:
            group = (
                ("author", target.author_id)
                if target.author_id is not None
                else ("source", target.message_id)
            )
        else:
            group = ("unaddressed", 0)
            target = None
        if group not in groups:
            groups[group] = (target, [])
        groups[group][1].append(mid)

    def branch_context(ids: list[int]) -> tuple[int, ...]:
        selected: set[int] = set()
        for mid in ids:
            seen = {mid}
            for _ in range(MAX_REPLY_DEPTH):
                target_id = nodes[mid].reply_to_message_id
                if target_id is None or target_id not in nodes or target_id in seen:
                    break
                seen.add(target_id)
                selected.add(target_id)
                mid = target_id
        return tuple(sorted(selected - set(ids)))

    branches = tuple(
        DialogueBranch(
            f"b{index}",
            target.author_id if target else None,
            target.author_name if target else None,
            target.author_username if target else None,
            tuple(ids),
            target is not None,
            branch_context(ids),
        )
        for index, (target, ids) in enumerate(groups.values(), 1)
    )
    return SharedDialogue(
        key[2],
        tuple(sorted(nodes.values(), key=lambda n: n.message_id)),
        branches,
        partial,
    )
