"""A bounded shared conversation, with explicit authors and reply edges."""

from dataclasses import dataclass, replace

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
    has_previous_summary: bool = False,
) -> SharedDialogue:
    """No names as keys, no cross-chat/topic ambient data, no future messages."""
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
    candidates: list[tuple[float, DialogueMessage]] = []
    ambient_references: dict[int, ReplyContext] = {}
    latest_recap = max(
        (max(e.message_ids) for e in own.entries if e.summary and max(e.message_ids) < message_id),
        default=0,
    )
    for other, session in memory.sessions.items():
        if len(other) != 3 or other[:2] != key[:2]:
            continue
        author = session.anchor
        for entry in session.entries:
            mid = max(entry.message_ids)
            if len(entry.message_ids) != 1 or mid in nodes:
                continue
            if (
                other == key
                and latest_recap
                and mid <= own.summarized_through
                and mid != latest_recap
            ):
                continue
            if not (0 <= sent_at - entry.sent_at <= memory.ttl and mid < message_id):
                continue
            ref = next((r for r in entry.reply_contexts if r.message_id == mid), None)
            if ref is not None:
                ambient_references[mid] = ref
            candidates.append(
                (
                    entry.sent_at,
                    DialogueMessage(
                        mid,
                        other[2],
                        author.author_name if author else None,
                        author.author_username if author else None,
                        "summary" if other == key and entry.summary else entry.kind,
                        (entry.summary or entry.text) if other == key else entry.text,
                        ref.reply_to_message_id if ref else None,
                    ),
                )
            )
    # Explicit sources are priority context, including a source across forum topics.
    for ref in references:
        if ref.reply_to_message_id not in nodes:
            text = ref.text[:budget]
            budget -= len(text)
            nodes[ref.reply_to_message_id] = DialogueMessage(
                ref.reply_to_message_id,
                ref.author_id,
                ref.author_name,
                ref.author_username,
                ref.kind,
                text,
                unavailable_reason=ref.unavailable_reason,
            )
    # First preserve available ancestors/descendants of focus reply chains.
    connected = set(nodes)
    remaining = sorted(candidates, key=lambda pair: (pair[0], pair[1].message_id), reverse=True)
    ordered: list[DialogueMessage] = []
    while remaining:
        selected = [
            pair
            for pair in remaining
            if (
                pair[1].message_id in connected
                or pair[1].reply_to_message_id in connected
                or any(n.reply_to_message_id == pair[1].message_id for n in nodes.values())
            )
        ]
        if not selected:
            break
        for pair in selected:
            remaining.remove(pair)
            node = pair[1]
            connected.add(node.message_id)
            if node.reply_to_message_id:
                connected.add(node.reply_to_message_id)
            ordered.append(node)
    ordered.extend(node for _, node in remaining)
    for node in ordered:
        if node.message_id in nodes:
            # A cached explicit source can gain its real author ID and reply edge.
            old = nodes[node.message_id]
            nodes[node.message_id] = replace(
                old,
                author_id=node.author_id,
                reply_to_message_id=node.reply_to_message_id,
                text=node.text if node.kind == "summary" else old.text,
                kind=node.kind if node.kind == "summary" else old.kind,
            )
            continue
        if budget <= 0 or len(nodes) >= 100:
            break
        text = node.text
        if len(text) > min(budget, 2000):
            text = text[: max(0, min(budget, 2000) - 1)] + "…"
        budget -= len(text)
        nodes[node.message_id] = replace(node, text=text)
        # A participant may reply to a bot/caption or an older source which was
        # never collected as ambient history. Keep that explicit edge intelligible.
        ref = ambient_references.get(node.message_id)
        if ref and ref.reply_to_message_id not in nodes and budget > 0 and len(nodes) < 100:
            text = ref.text[: min(budget, 2000)]
            budget -= len(text)
            nodes[ref.reply_to_message_id] = DialogueMessage(
                ref.reply_to_message_id,
                ref.author_id,
                ref.author_name,
                ref.author_username,
                ref.kind,
                text,
                unavailable_reason=ref.unavailable_reason,
            )

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
    branches = tuple(
        DialogueBranch(
            f"b{index}",
            target.author_id if target else None,
            target.author_name if target else None,
            target.author_username if target else None,
            tuple(ids),
            target is not None,
        )
        for index, (target, ids) in enumerate(groups.values(), 1)
    )
    return SharedDialogue(
        key[2],
        tuple(sorted(nodes.values(), key=lambda n: n.message_id)),
        branches,
        partial,
        has_previous_summary,
    )
