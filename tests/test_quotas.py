import asyncio
import json
import stat
from datetime import date
from types import SimpleNamespace

import pytest

from voice_bot.quotas import PrivateVoiceQuota


def message(chat=123, *, kind="private"):
    return SimpleNamespace(chat=SimpleNamespace(id=chat, type=kind))


async def test_concurrent_private_requests_cannot_exceed_daily_limit(tmp_path):
    path = tmp_path / "usage.json"
    quota = PrivateVoiceQuota(path, 30)
    decisions = await asyncio.gather(*(quota.admit(message()) for _ in range(50)))
    assert decisions.count("allowed") == 30
    assert decisions.count("notify") == 1
    assert decisions.count("limited") == 19
    assert list(quota.users.values())[0].used == 30
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert all(len(key) == 64 for key in json.loads(path.read_text()))


async def test_quota_and_one_time_notice_survive_restart(tmp_path):
    path = tmp_path / "usage.json"
    quota = PrivateVoiceQuota(path, 1)
    assert await quota.admit(message()) == "allowed"
    assert await PrivateVoiceQuota(path, 1).admit(message()) == "notify"
    assert await PrivateVoiceQuota(path, 1).admit(message()) == "limited"


async def test_quota_resets_on_next_utc_day_and_is_independent_between_users(tmp_path):
    today = [date(2026, 1, 1)]
    quota = PrivateVoiceQuota(tmp_path / "usage.json", 1, clock=lambda: today[0])
    assert await quota.admit(message(123)) == "allowed"
    assert await quota.admit(message(123)) == "notify"
    assert await quota.admit(message(456)) == "allowed"
    today[0] = date(2026, 1, 2)
    assert await quota.admit(message(123)) == "allowed"
    assert len(quota.users) == 1
    assert await quota.admit(message(123)) == "notify"


async def test_groups_do_not_consume_private_allowance_or_create_state(tmp_path):
    path = tmp_path / "usage.json"
    quota = PrivateVoiceQuota(path, 1)
    for _ in range(40):
        assert await quota.admit(message(-100, kind="supergroup")) == "allowed"
    assert not quota.users and not path.exists()


async def test_capacity_never_evicts_an_existing_daily_counter(tmp_path):
    quota = PrivateVoiceQuota(tmp_path / "usage.json", 1, max_users=2)
    assert await quota.admit(message(1)) == "allowed"
    assert await quota.admit(message(2)) == "allowed"
    assert await quota.admit(message(3)) == "limited"
    assert await quota.admit(message(1)) == "notify"


async def test_rejected_queue_submission_can_refund_its_allowance(tmp_path):
    quota = PrivateVoiceQuota(tmp_path / "usage.json", 1)
    assert await quota.admit(message()) == "allowed"
    await quota.refund(message())
    await quota.refund(message())
    assert list(quota.users.values())[0].used == 0
    assert await quota.admit(message()) == "allowed"


async def test_failed_storage_does_not_open_unlimited_private_processing(tmp_path, monkeypatch):
    quota = PrivateVoiceQuota(tmp_path / "usage.json", 30)

    def fail_write(users):
        raise OSError("disk unavailable")

    monkeypatch.setattr(quota, "_write", fail_write)
    with pytest.raises(OSError):
        await quota.admit(message())
    assert not quota.available
    assert await quota.admit(message()) == "limited"
    assert not quota.users


@pytest.mark.parametrize("content", ["invalid", '{"raw-user-id": {}}'])
def test_invalid_saved_quota_fails_closed(tmp_path, content):
    path = tmp_path / "usage.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        PrivateVoiceQuota(path, 30)
    assert path.read_text() == content
