import asyncio
import io
from collections.abc import Buffer
from pathlib import Path

from voice_bot.errors import UserFacingError


class BoundedFile(io.BufferedWriter):
    """Abort a Telegram download before it can exceed the actual on-disk limit."""

    def __init__(self, path: Path, limit: int) -> None:
        super().__init__(io.FileIO(path, "wb"))
        self.limit = limit

    def write(self, data: Buffer, /) -> int:
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise UserFacingError("Голосовое слишком большое для обработки.")
        return super().write(data)


async def convert_audio(
    source: Path, destination: Path, *, binary: str, max_bytes: int, timeout_seconds: float
) -> None:
    """Convert OGG/Opus to mono PCM WAV without trimming."""
    try:
        process = await asyncio.create_subprocess_exec(
            binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-fs",
            str(max_bytes + 1),
            str(destination),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (OSError, ValueError):
        raise UserFacingError("Конвертер аудио недоступен. Сообщите владельцу бота.") from None
    try:
        async with asyncio.timeout(timeout_seconds):
            await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode != 0:
        raise UserFacingError("Не удалось прочитать аудио. Попробуйте другое голосовое.")
    try:
        size = (await asyncio.to_thread(destination.stat)).st_size
    except FileNotFoundError:
        size = 0
    if size <= 44:
        raise UserFacingError("В голосовом не найдено аудио.")
    if size >= max_bytes:
        raise UserFacingError("Аудио слишком большое после конвертации; отправьте короче.")
