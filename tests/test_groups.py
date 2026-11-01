import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from voice_bot.groups import GroupStore


def test_decisions_survive_restart_and_unknown_groups_fail_closed(tmp_path):
    path = tmp_path / "groups.json"
    store = GroupStore(path)
    assert not store.allowed(-100)
    group, joined = store.observe(-100, "Тест <группа>")
    assert joined and group.status == "pending"
    assert not store.allowed(-100)
    store.decide(-100, group.revision, "allowed")
    assert GroupStore(path).allowed(-100)
    group = store.groups[-100]
    store.decide(-100, group.revision, "denied")
    assert not GroupStore(path).allowed(-100)
    assert path.stat().st_mode & 0o777 == 0o600
    assert set(json.loads(path.read_text())[0]) == {
        "chat_id",
        "title",
        "status",
        "present",
        "revision",
        "notified",
    }


def test_removal_and_rejoin_require_new_approval_and_invalidate_old_buttons(tmp_path):
    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    allowed = store.decide(-100, group.revision, "allowed")
    store.leave(-100)
    assert not store.allowed(-100)
    group, joined = store.observe(-100, "New title")
    assert joined and group.status == "pending"
    assert group.title == "New title"
    assert not group.notified
    assert store.decide(-100, allowed.revision, "allowed") is None


def test_duplicate_observation_does_not_reset_decision_or_notification(tmp_path):
    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    store.mark_notified(group)
    group = store.groups[-100]
    store.decide(-100, group.revision, "denied")
    group, joined = store.observe(-100, "Renamed")
    assert not joined and group.status == "denied" and group.notified


def test_corrupted_permissions_refuse_to_start(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text('[{"chat_id": -100, "title": "Test", "status": "anything"}]')
    with pytest.raises(ValidationError):
        GroupStore(path)


def test_disk_failure_does_not_grant_permission(tmp_path):
    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    with patch("voice_bot.groups.os.replace", side_effect=OSError):
        with pytest.raises(OSError):
            store.decide(-100, group.revision, "allowed")
    assert not store.allowed(-100)
    assert not GroupStore(store.path).allowed(-100)
    assert list(tmp_path.iterdir()) == [store.path]


def test_repeating_same_decision_does_not_rewrite_state_or_increment_revision(tmp_path):
    store = GroupStore(tmp_path / "groups.json")
    group, _ = store.observe(-100, "Test")
    group = store.decide(-100, group.revision, "allowed")
    before = store.path.read_bytes()
    with patch("voice_bot.groups.os.replace") as replace:
        same = store.decide(-100, group.revision, "allowed")
    replace.assert_not_called()
    assert same == group and store.path.read_bytes() == before
