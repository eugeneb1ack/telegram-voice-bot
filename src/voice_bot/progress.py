"""Best-effort activity indicator; never delays or fails the actual job."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from aiogram import Bot

logger = logging.getLogger(__name__)


@asynccontextmanager
async def typing(
    bot: Bot,
    chat_id: int,
    topic_id: int | None = None,
    *,
    enabled: Callable[[], bool] = lambda: True,
    interval: float = 4,
) -> AsyncIterator[None]:
    async def pulse() -> None:
        announced = False
        while enabled():
            try:
                async with asyncio.timeout(3):
                    await bot.send_chat_action(
                        chat_id=chat_id, message_thread_id=topic_id, action="typing"
                    )
                if not announced:
                    logger.info("typing_started chat_id=%s topic_id=%s", chat_id, topic_id or 0)
                    announced = True
            except Exception as exc:
                logger.debug("typing_unavailable error_type=%s", type(exc).__name__)
                return
            await asyncio.sleep(interval)

    task = asyncio.create_task(pulse(), name="voice-typing")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
