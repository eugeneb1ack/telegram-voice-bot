from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    telegram_bot_token: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.openai.com/v1"
    telegram_proxy_url: SecretStr = SecretStr("")
    openai_proxy_url: SecretStr = SecretStr("")
    stt_model: str = "gpt-transcribe"
    stt_language: str = ""
    summary_model: str = "gpt-6-luna"
    summary_language: str = "русский"
    summary_max_output_tokens: int = Field(default=1200, ge=128, le=10000)
    summary_max_input_chars: int = Field(default=100000, ge=1000, le=1000000)
    summary_short_input_words: int = Field(default=80, ge=1, le=10000)
    summary_long_input_words: int = Field(default=300, ge=1, le=100000)
    summary_batch_detail_min_words: int = Field(default=180, ge=1, le=100000)
    summary_short_target_words: int = Field(default=40, ge=1, le=2000)
    summary_medium_target_words: int = Field(default=80, ge=1, le=2000)
    summary_long_target_words: int = Field(default=140, ge=1, le=2000)
    allowed_chat_ids: set[int] = Field(default_factory=set)
    owner_id: int = Field(default=0, ge=0)
    group_state_file: Path = Path("data/groups.json")
    queue_state_file: Path = Path("data/queue.sqlite3")
    processed_state_file: Path = Path("data/processed.jsonl")
    processed_max_messages: int = Field(default=50000, ge=100, le=1000000)
    private_usage_file: Path = Path("data/private-usage.json")
    private_daily_voice_limit: int = Field(default=30, ge=1, le=1000)
    summary_min_voice_seconds: int = Field(default=30, ge=0, le=600)
    context_ttl_seconds: int = Field(default=900, ge=0, le=3600)
    context_max_chars: int = Field(default=12000, ge=100, le=50000)
    context_max_sessions: int = Field(default=200, ge=1, le=2000)
    context_max_messages: int = Field(default=12, ge=2, le=50)
    context_state_file: Path = Path("data/conversations.json")
    context_summary_every: int = Field(default=3, ge=0, le=10)
    context_series_gap_seconds: int = Field(default=900, ge=10, le=900)
    voice_batch_pause_seconds: float = Field(default=3, ge=0, le=5)
    voice_batch_max_messages: int = Field(default=10, ge=2, le=10)
    max_voice_seconds: int = Field(default=600, ge=1, le=600)
    max_download_bytes: int = Field(default=19000000, ge=1024, le=20000000)
    max_upload_bytes: int = Field(default=20000000, ge=1024, le=24000000)
    max_concurrent_jobs: int = Field(default=4, ge=1, le=20)
    max_pending_jobs: int = Field(default=100, ge=1, le=1000)
    max_pending_jobs_per_chat: int = Field(default=20, ge=1, le=1000)
    max_pending_voices_per_user: int = Field(default=12, ge=1, le=1000)
    request_timeout_seconds: float = Field(default=90, gt=0, le=180)
    job_timeout_seconds: float = Field(default=300, gt=0, le=900)
    download_timeout_seconds: int = Field(default=45, ge=1, le=120)
    conversion_timeout_seconds: float = Field(default=45, gt=0, le=120)
    api_max_attempts: int = Field(default=3, ge=1, le=5)
    telegram_max_retry_after_seconds: int = Field(default=30, ge=1, le=120)
    ffmpeg_binary: str = "ffmpeg"

    @field_validator("telegram_proxy_url", "openai_proxy_url")
    @classmethod
    def validate_proxy(cls, value: SecretStr) -> SecretStr:
        url = value.get_secret_value()
        if not url:
            return value
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise ValueError("Invalid proxy URL") from None
        if (
            parts.scheme not in {"http", "socks5"}
            or not parts.hostname
            or port == 0
            or parts.path not in {"", "/"}
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Proxy URL must use HTTP CONNECT or SOCKS5 without a path or query")
        return value

    @field_validator("openai_base_url")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("API base URL must be HTTPS without credentials, query or fragment")
        return value.rstrip("/")

    def missing_credentials(self) -> list[str]:
        missing = []
        if not self.telegram_bot_token.get_secret_value().strip():
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.openai_api_key.get_secret_value().strip():
            missing.append("OPENAI_API_KEY")
        return missing
