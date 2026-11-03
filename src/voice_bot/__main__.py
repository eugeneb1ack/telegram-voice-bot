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
from aiogram.types import BotCommand, BotCommandScopeAllGroupChats, BotCommandScopeChat
from pydantic import ValidationError

from voice_bot.admin import GroupAdmin, build_admin_router
from voice_bot.commands import CommandCleanup
from voice_bot.config import Settings
from voice_bot.groups import GroupStore
from voice_bot.handlers import build_router
from voice_bot.journal import JournalUnavailable, QueueJournal
from voice_bot.modes import build_modes_router
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
                BotCommand(
                    command="text", description="Расшифровать только для меня", is_ephemeral=True
                ),
            ]
        )
        await bot.set_my_commands(
            [
                BotCommand(
                    command="text", description="Расшифровать только для меня", is_ephemeral=True
                ),
                BotCommand(
                    command="settings", description="Выбрать режим работы", is_ephemeral=True
                ),
                BotCommand(command="start", description="О боте"),
            ],
            scope=BotCommandScopeAllGroupChats(),
        )
        try:
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description="Добавить бота в группу"),
                    BotCommand(
                        command="text",
                        description="Расшифровать только для меня",
                        is_ephemeral=True,
                    ),
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
            limits=httpx.Limits(
                max_connections=settings.max_concurrent_jobs * 2,
                # Voice series often pause longer than HTTPX's five-second default.
                # Reuse the proxy/TLS connection instead of reconnecting for each voice.
                keepalive_expiry=60,
            ),
            trust_env=False,
        ) as client:
            api = APIClient(client, settings)
            journal = QueueJournal(
                settings.queue_state_file,
                max_jobs=settings.max_pending_jobs * (settings.voice_batch_max_messages + 1)
                + settings.max_concurrent_jobs,
                max_receipts=settings.processed_max_messages,
            )
            service = VoiceService(
                settings, HTTPTranscriber(api), HTTPSummarizer(api), admin=admin, journal=journal
            )
            dispatcher = Dispatcher()
            cleanup = CommandCleanup(max_pending=settings.max_pending_jobs)
            dispatcher.message.middleware(cleanup)
            dispatcher.include_router(build_admin_router(admin))
            dispatcher.include_router(build_modes_router(admin))
            dispatcher.include_router(build_router(service))

            async def guard_polling(make_request: Any, bot: Bot, method: Any) -> Any:
                if isinstance(method, GetUpdates) and journal.failed.is_set():
                    raise JournalUnavailable("Queue journal unavailable")
                return await make_request(bot, method)

            async def stop_on_storage_failure() -> None:
                await journal.failed.wait()
                logger.error("queue_storage_unavailable")
                await dispatcher.stop_polling()

            session.middleware(guard_polling)
            await admin.notify_pending(bot)
            service.start(bot)
            logger.info(
                "polling_started bot_id=%s username=%s",
                me.id,
                me.username,
            )
            # Update handlers enqueue work; voice workers and pending jobs are bounded separately.
            watcher = asyncio.create_task(stop_on_storage_failure())
            try:
                await service.recover(bot)
                await dispatcher.start_polling(
                    bot,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    handle_as_tasks=False,  # Persist admission before getUpdates advances offset.
                    close_bot_session=False,
                )
                if journal.failed.is_set():
                    raise JournalUnavailable("Queue journal unavailable")
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                try:
                    await service.close()
                finally:
                    await cleanup.close()
                    journal.close()


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
