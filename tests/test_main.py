from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramForbiddenError
from aiogram.methods import SetMyCommands
from aiogram.types import User
from pydantic import SecretStr

from voice_bot.__main__ import run


@pytest.fixture
def startup(settings, tmp_path, monkeypatch):
    settings.owner_id = 987654321
    settings.group_state_file = tmp_path / "groups.json"
    bot = MagicMock()
    bot.__aenter__ = AsyncMock(return_value=bot)
    bot.__aexit__ = AsyncMock(return_value=None)
    bot.me = AsyncMock(return_value=User(id=123456, is_bot=True, first_name="Voice"))
    bot.get_webhook_info = AsyncMock(return_value=SimpleNamespace(url=""))
    bot.set_my_commands = AsyncMock()
    bot.send_message = AsyncMock()
    dispatcher = MagicMock()
    dispatcher.start_polling = AsyncMock()
    monkeypatch.setattr("voice_bot.__main__.Bot", MagicMock(return_value=bot))
    monkeypatch.setattr("voice_bot.__main__.Dispatcher", MagicMock(return_value=dispatcher))
    return bot, dispatcher


async def test_owner_not_started_bot_does_not_block_polling(settings, startup, caplog):
    bot, dispatcher = startup
    error = TelegramForbiddenError(
        method=SetMyCommands(commands=[]), message="SECRET Telegram request"
    )
    bot.set_my_commands.side_effect = [True, error]
    bot.send_message.side_effect = error
    await run(settings)
    dispatcher.start_polling.assert_awaited_once()
    assert "SECRET" not in caplog.text


async def test_existing_webhook_is_not_removed_or_polled(settings, startup):
    bot, dispatcher = startup
    bot.get_webhook_info.return_value = SimpleNamespace(url="https://example.com/private")
    with pytest.raises(RuntimeError):
        await run(settings)
    bot.delete_webhook.assert_not_called()
    dispatcher.start_polling.assert_not_awaited()
    bot.set_my_commands.assert_not_awaited()


async def test_admin_is_not_published_in_bot_command_menus(settings, startup):
    bot, _ = startup
    await run(settings)
    assert bot.set_my_commands.await_count == 2
    for call in bot.set_my_commands.await_args_list:
        assert {command.command for command in call.args[0]} == {"start", "text"}


@pytest.mark.parametrize(
    "telegram_proxy,openai_proxy",
    [
        ("", ""),
        ("http://telegram-proxy:8080", "http://openai-proxy:8080"),
        ("socks5://telegram-proxy:1080", "socks5://openai-proxy:1080"),
    ],
)
async def test_telegram_and_openai_use_independent_proxies(
    settings, startup, monkeypatch, telegram_proxy, openai_proxy
):
    import httpx
    from aiogram.client.session.aiohttp import AiohttpSession

    settings.telegram_proxy_url = SecretStr(telegram_proxy)
    settings.openai_proxy_url = SecretStr(openai_proxy)
    telegram_session = MagicMock(wraps=AiohttpSession)
    api_client = MagicMock(wraps=httpx.AsyncClient)
    monkeypatch.setattr("voice_bot.__main__.AiohttpSession", telegram_session)
    monkeypatch.setattr("voice_bot.__main__.httpx.AsyncClient", api_client)
    await run(settings)
    telegram_session.assert_called_once_with(timeout=60, proxy=telegram_proxy or None)
    assert api_client.call_args.kwargs["proxy"] == (openai_proxy or None)
    assert api_client.call_args.kwargs["trust_env"] is False
    startup[1].start_polling.assert_awaited_once()
