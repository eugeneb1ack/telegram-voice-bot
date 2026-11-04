import asyncio
import shutil
import wave
from unittest.mock import AsyncMock, Mock

import pytest

from voice_bot.audio import convert_audio
from voice_bot.errors import UserFacingError


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="optional local FFmpeg smoke test")
async def test_real_ffmpeg_converts_local_synthetic_audio(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "output.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(b"\0" * (4800 * 2 * 2))
    await convert_audio(source, destination, binary="ffmpeg", max_bytes=200000, timeout_seconds=5)
    with wave.open(str(destination), "rb") as converted:
        assert converted.getnchannels() == 1
        assert converted.getframerate() == 16000
        assert 1500 <= converted.getnframes() <= 1700


async def test_missing_ffmpeg_returns_safe_error(tmp_path):
    with pytest.raises(UserFacingError, match="Конвертер аудио недоступен"):
        await convert_audio(
            tmp_path / "source.ogg",
            tmp_path / "out.wav",
            binary="/nonexistent-voice-bot-test-ffmpeg",
            max_bytes=1000,
            timeout_seconds=1,
        )


async def test_failed_ffmpeg_does_not_expose_stderr(tmp_path, monkeypatch):
    process = type("Process", (), {"returncode": 1, "wait": AsyncMock(return_value=1)})()
    monkeypatch.setattr(
        "voice_bot.audio.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    )
    with pytest.raises(UserFacingError, match="Не удалось прочитать аудио"):
        await convert_audio(
            tmp_path / "source.ogg",
            tmp_path / "out.wav",
            binary="ffmpeg",
            max_bytes=1000,
            timeout_seconds=1,
        )


async def test_ffmpeg_timeout_kills_and_reaps_process(tmp_path, monkeypatch):
    process = Mock(returncode=None)

    async def wait():
        if process.returncode is None:
            await asyncio.Event().wait()
        return process.returncode

    def kill():
        process.returncode = -9

    process.wait = AsyncMock(side_effect=wait)
    process.kill.side_effect = kill
    monkeypatch.setattr(
        "voice_bot.audio.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    )
    with pytest.raises(TimeoutError):
        await convert_audio(
            tmp_path / "source.ogg",
            tmp_path / "out.wav",
            binary="ffmpeg",
            max_bytes=1000,
            timeout_seconds=0.01,
        )
    process.kill.assert_called_once()
    assert process.wait.await_count == 2
