import pytest
from pydantic import SecretStr, ValidationError


def test_one_openai_key_powers_both_stages(settings):
    assert not settings.missing_credentials()
    settings.openai_api_key = SecretStr("")
    assert settings.missing_credentials() == ["OPENAI_API_KEY"]
    settings.telegram_bot_token = SecretStr("")
    assert settings.missing_credentials() == ["TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/v1",
        "https://token@example.com/v1",
        "https://example.com/?key=SECRET",
    ],
)
def test_unsafe_endpoints_are_rejected(settings, endpoint):
    with pytest.raises(ValidationError) as exc:
        type(settings)(_env_file=None, openai_base_url=endpoint)
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize("field", ["telegram_proxy_url", "openai_proxy_url"])
@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com:1080",
        "http://",
        "http://user:SECRET@example.com:bad",
        "http://example.com:0",
        "http://example.com/path?key=SECRET",
    ],
)
def test_invalid_proxies_are_rejected_without_credentials(settings, field, url):
    with pytest.raises(ValidationError) as exc:
        type(settings)(_env_file=None, **{field: url})
    assert "SECRET" not in str(exc.value)


def test_proxy_credentials_are_redacted(settings):
    settings.telegram_proxy_url = SecretStr("http://user:SECRET@example.com:8080")
    settings.openai_proxy_url = SecretStr("socks5://user:SECRET@example.net:1080")
    assert "SECRET" not in repr(settings)
