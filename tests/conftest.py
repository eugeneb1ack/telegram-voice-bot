import pytest
from pydantic import SecretStr

from voice_bot.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("123456:test-token"),
        openai_api_key=SecretStr("test-openai-key"),
        api_max_attempts=1,
        allowed_chat_ids=set(),
        processed_state_file=tmp_path / "processed.jsonl",
        private_usage_file=tmp_path / "private-usage.json",
    )
