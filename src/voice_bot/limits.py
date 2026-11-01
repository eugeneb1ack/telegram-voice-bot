"""Sender identity and bounded throttling for service notices."""

import time
from collections import OrderedDict
from collections.abc import Callable

from aiogram.types import Message

SenderKey = tuple[int, str, int]


def sender_key(message: Message) -> SenderKey | None:
    chat = getattr(message, "sender_chat", None)
    user = getattr(message, "from_user", None)
    if chat is not None:
        return message.chat.id, "chat", chat.id
    if user is not None:
        return message.chat.id, "user", user.id
    return None


class NoticeLimiter:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        *,
        sender_interval: float = 30,
        chat_interval: float = 5,
    ) -> None:
        self.clock = clock
        self.sender_interval, self.chat_interval = sender_interval, chat_interval
        self.senders: OrderedDict[SenderKey, float] = OrderedDict()
        self.chats: OrderedDict[int, float] = OrderedDict()

    def allow(self, message: Message) -> bool:
        now = self.clock()
        for entries in (self.senders, self.chats):
            while entries and next(iter(entries.values())) <= now:
                entries.popitem(last=False)
        key = sender_key(message) or (message.chat.id, "unknown", 0)
        if key in self.senders or message.chat.id in self.chats:
            return False
        self.senders[key] = now + self.sender_interval
        self.chats[message.chat.id] = now + self.chat_interval
        for entries in (self.senders, self.chats):
            while len(entries) > 2048:
                entries.popitem(last=False)
        return True
