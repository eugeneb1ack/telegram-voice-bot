import argparse
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramAPIError
from aiogram.methods import GetUpdates
from aiogram.types import BotCommand, BotCommandScopeChat
from pydantic import ValidationError

from voice_bot.admin import GroupAdmin, build_admin_router
from voice_bot.config import Settings
from voice_bot.groups import GroupStore
from voice_bot.handlers import build_router
from voice_bot.providers import APIClient, HTTPSummarizer, HTTPTranscriber
from voice_bot.service import VoiceService

logger = logging.getLogger(__name__)


async def polling_health(make_request: Any, bot: Bot, method: Any) -> Any:
    result = await make_request(bot, method)
    if isinstance(method, GetUpdates):
        await asyncio.to_thread(Path("/tmp/voice-bot-polling").touch)
    return result


async def run(settings: Settings) -> None:
    # Suppress third-party request/response logging even when a host config enables it.
    for name in ("aiogram", "httpx", "httpcore", "aiohttp"):
        logging.getLogger(name).disabled = True
    session = AiohttpSession(
        timeout=60, proxy=settings.telegram_proxy_url.get_secret_value() or None
    )
    session.middleware(polling_health)
    store = GroupStore(settings.group_state_file)
    admin = GroupAdmin(settings, store)
    async with Bot(token=settings.telegram_bot_token.get_secret_value(), session=session) as bot:
        me = await bot.me()
        webhook = await bot.get_webhook_info()
        if webhook.url:
            raise RuntimeError("Webhook is configured; refusing to remove it automatically")
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Добавить бота в группу"),
                BotCommand(command="text", description="Расшифровать голосовое в ответе"),
            ]
        )
        try:
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description="Добавить бота в группу"),
                    BotCommand(command="text", description="Расшифровать голосовое в ответе"),
                ],
                scope=BotCommandScopeChat(chat_id=settings.owner_id),
            )
        except TelegramAPIError:
            # The owner may not have opened the bot yet. Group approvals stay closed.
            logger.warning("owner_command_menu_unavailable")
        async with httpx.AsyncClient(
            proxy=settings.openai_proxy_url.get_secret_value() or None,
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=settings.max_concurrent_jobs * 2),
            trust_env=False,
        ) as client:
            api = APIClient(client, settings)
            service = VoiceService(settings, HTTPTranscriber(api), HTTPSummarizer(api), admin=admin)
            dispatcher = Dispatcher()
            dispatcher.include_router(build_admin_router(admin))
            dispatcher.include_router(build_router(service))
            await admin.notify_pending(bot)
            logger.info(
                "polling_started bot_id=%s username=%s",
                me.id,
                me.username,
            )
            # Update handlers enqueue work; voice workers and pending jobs are bounded separately.
            try:
                await dispatcher.start_polling(
                    bot,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    tasks_concurrency_limit=settings.max_concurrent_jobs + 4,
                    close_bot_session=False,
                )
            finally:
                await service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Bot: Telegram voice summaries")
    parser.add_argument(
        "--check", action="store_true", help="validate local setup; no network calls"
    )
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        parser.exit(2, f"Некорректные настройки: {fields}. Значения скрыты.\n")
    missing = settings.missing_credentials()
    if not settings.owner_id:
        missing.append("OWNER_ID")
    if missing:
        parser.exit(2, "Заполните .env: " + ", ".join(missing) + "\n")
    if not shutil.which(settings.ffmpeg_binary):
        parser.exit(2, "FFmpeg не найден. Установите его или задайте FFMPEG_BINARY.\n")
    if settings.summary_min_voice_seconds > settings.max_voice_seconds:
        parser.exit(2, "SUMMARY_MIN_VOICE_SECONDS не может превышать MAX_VOICE_SECONDS.\n")
    if args.check:
        print("Локальные настройки проверены. Ключи и доступ к моделям по сети не проверялись.")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: record.name.startswith("voice_bot"))
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        parser.exit(1, f"Не удалось запустить бота ({type(exc).__name__}). Значения скрыты.\n")


if __name__ == "__main__":
    main()
