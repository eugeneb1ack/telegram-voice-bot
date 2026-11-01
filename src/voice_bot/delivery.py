"""Native per-user replies inside group chats."""

from datetime import UTC, datetime

from aiogram.types import EphemeralMessageParameters, Message, ReplyParameters


def personal(message: Message) -> Message | None:
    if message.chat.type not in {"group", "supergroup"}:
        return message
    if message.sender_chat or not message.from_user or message.from_user.is_bot:
        return None
    return message.model_copy(update={"receiver_user": message.from_user})


def recipient(message: Message) -> int | None:
    user = getattr(message, "receiver_user", None)
    return user.id if user is not None else None


def ephemeral(message: Message) -> EphemeralMessageParameters | None:
    user_id = recipient(message)
    return EphemeralMessageParameters(receiver_user_id=user_id) if user_id else None


def reply_to(message: Message) -> ReplyParameters | None:
    if message.message_id and (not recipient(message) or getattr(message, "voice", None)):
        return ReplyParameters(message_id=message.message_id, allow_sending_without_reply=False)
    ephemeral_id = getattr(message, "ephemeral_message_id", None)
    if ephemeral_id and (datetime.now(UTC) - message.date).total_seconds() < 14:
        return ReplyParameters(ephemeral_message_id=ephemeral_id)
    # Queued work can outlive the command's reply window; bot admins can send without it.
    return None
