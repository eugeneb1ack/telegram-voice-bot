"""Persistent group access settings."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

GroupStatus = Literal["pending", "allowed", "denied"]


class Group(BaseModel, frozen=True):
    chat_id: int = Field(lt=0)
    title: str
    status: GroupStatus = "pending"
    present: bool = True
    revision: int = Field(default=1, ge=1)
    notified: bool = False


class GroupStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.groups: dict[int, Group] = {}
        if path.exists():
            groups = TypeAdapter(list[Group]).validate_json(path.read_bytes())
            self.groups = {group.chat_id: group for group in groups}

    def _save(self, group: Group) -> Group:
        groups = self.groups | {group.chat_id: group}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump([g.model_dump() for g in groups.values()], file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # A failed disk write must never grant access in memory.
        self.groups = groups
        return group

    def observe(self, chat_id: int, title: str) -> tuple[Group, bool]:
        group = self.groups.get(chat_id)
        joined = group is None or not group.present
        if group is None:
            group = self._save(Group(chat_id=chat_id, title=title))
        elif not group.present:
            group = self._save(
                group.model_copy(
                    update={
                        "title": title,
                        "status": "pending",
                        "present": True,
                        "notified": False,
                        "revision": group.revision + 1,
                    }
                )
            )
        elif group.title != title:
            group = self._save(group.model_copy(update={"title": title}))
        return group, joined

    def leave(self, chat_id: int) -> None:
        group = self.groups.get(chat_id)
        if group and group.present:
            self._save(group.model_copy(update={"present": False, "revision": group.revision + 1}))

    def allowed(self, chat_id: int) -> bool:
        group = self.groups.get(chat_id)
        return bool(group and group.present and group.status == "allowed")

    def decide(self, chat_id: int, revision: int, status: GroupStatus) -> Group | None:
        group = self.groups.get(chat_id)
        if group is None or not group.present or group.revision != revision:
            return None
        if group.status == status:
            return group
        return self._save(
            group.model_copy(update={"status": status, "revision": group.revision + 1})
        )

    def mark_notified(self, group: Group) -> None:
        current = self.groups.get(group.chat_id)
        if current == group:
            self._save(group.model_copy(update={"notified": True}))
