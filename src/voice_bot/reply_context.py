"""Bounded, explicitly referenced messages; separate from the author's own words."""

from dataclasses import dataclass, replace

from aiogram.types import Message


@dataclass(frozen=True)
class PendingReplyMedia:
    file_id: str
    file_unique_id: str
    duration: int
    file_size: int | None
    date: float


@dataclass(frozen=True)
class ReplyContext:
    message_id: int
    reply_to_message_id: int
    author_name: str | None
    author_username: str | None
    kind: str
    text: str
    unavailable_reason: str | None = None
    author_id: int | None = None
    pending_media: PendingReplyMedia | None = None


def reply_target(message: Message) -> Message | None:
    target = getattr(message, "reply_to_message", None)
    if target is None:
        external = getattr(message, "external_reply", None)
        if (
            external is None
            or external.chat is None
            or external.chat.id != message.chat.id
            or not external.message_id
        ):
            return None
        target = Message(
            message_id=external.message_id,
            date=external.origin.date,
            chat=external.chat,
            forward_origin=external.origin,
            voice=external.voice,
            video_note=external.video_note,
        )
    if (
        getattr(getattr(target, "chat", None), "id", None) != message.chat.id
        or getattr(target, "message_id", 0) <= 0
        or getattr(target, "receiver_user", None) is not None
        or getattr(target, "ephemeral_message_id", None)
        or not any(
            getattr(target, field, None) is not None
            for field in ("from_user", "sender_chat", "forward_origin")
        )
    ):
        return None
    return target


def media_kind(message: Message) -> str | None:
    for kind in ("voice", "video_note"):
        if getattr(message, kind, None) is not None:
            return kind
    return "text" if getattr(message, "text", None) or getattr(message, "caption", None) else None


def as_voice(message: Message) -> Message | None:
    if getattr(message, "voice", None) is not None and not getattr(message, "video_note", None):
        return message
    return None


def bounded_contexts(
    contexts: tuple[ReplyContext, ...], max_chars: int
) -> tuple[ReplyContext, ...]:
    return tuple(
        replace(context, kind="recipient", text="", unavailable_reason=None, pending_media=None)
        for context in contexts[:100]
    )
