import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import ErrorEvent, Message

from voice_bot.delivery import personal
from voice_bot.modes import is_admin
from voice_bot.service import VoiceService

logger = logging.getLogger(__name__)


def build_router(service: VoiceService) -> Router:
    router = Router(name="voice")

    @router.message(Command("text"), flags={"cleanup_command": True})
    async def transcribe_reply(message: Message, bot: Bot) -> None:
        target = personal(message)
        if target is None:
            return
        message = target
        logger.info(
            "text_command_received chat_id=%s ephemeral=%s",
            message.chat.id,
            bool(message.ephemeral_message_id),
        )
        if service.admin is not None and message.chat.type in {"group", "supergroup"}:
            await service.admin.observe(message.chat, bot)
        if not service.chat_allowed(message):
            return
        original = message.reply_to_message
        if original is None or not original.voice:
            external = message.external_reply
            if (
                external is not None
                and external.voice is not None
                and external.chat is not None
                and external.chat.id == message.chat.id
                and external.message_id is not None
            ):
                # Telegram uses external_reply for references across forum topics.
                # Reply to the original voice in the same approved chat, never another chat.
                original = Message(
                    message_id=external.message_id,
                    date=external.origin.date,
                    chat=external.chat,
                    voice=external.voice,
                    forward_origin=external.origin,
                ).as_(bot)
            else:
                if original is None and external is None:
                    text = "Отправьте /text ответом на голосовое сообщение."
                elif external is not None:
                    text = (
                        "Telegram не передал доступный оригинал голосового. "
                        "Перешлите саму запись в этот чат или мне в личку."
                    )
                else:
                    text = (
                        "В сообщении, на которое вы ответили, нет голосового. "
                        "Ответьте /text на саму запись или перешлите её мне в личку."
                    )
                await service.notify(message, text)
                return
        if original.chat.id != message.chat.id:
            await service.notify(message, "Перешлите саму запись в этот чат или мне в личку.")
            return
        if message.receiver_user is not None:
            if not await is_admin(bot, message.chat.id, bot.id):
                await service.notify(
                    message, "Для личных расшифровок в группе назначьте бота администратором."
                )
                return
            original = original.model_copy(update={"receiver_user": message.receiver_user})
        await service.enqueue(original, bot, requester=message)

    router.message.register(service.accept, F.voice)
    router.message.register(service.observe_text, F.text)

    @router.errors()
    async def safe_error_handler(event: ErrorEvent) -> bool:
        logger.warning("update_failed error_type=%s", type(event.exception).__name__)
        return True

    return router
