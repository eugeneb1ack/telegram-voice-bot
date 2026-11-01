from types import SimpleNamespace

from voice_bot.limits import NoticeLimiter, sender_key


def message(user=1, chat=-100):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat),
        from_user=SimpleNamespace(id=user),
    )


def test_notices_are_bounded_per_sender_and_chat_without_delaying_results():
    now = [0.0]
    limiter = NoticeLimiter(clock=lambda: now[0])
    assert limiter.allow(message())
    assert not limiter.allow(message(user=2))
    assert limiter.allow(message(chat=-200))
    now[0] = 5
    assert limiter.allow(message(user=2))
    assert not limiter.allow(message())
    now[0] = 30
    assert limiter.allow(message())


def test_notice_state_is_bounded_for_many_senders():
    limiter = NoticeLimiter()
    for index in range(3000):
        assert limiter.allow(message(user=index, chat=-index))
    assert len(limiter.senders) == len(limiter.chats) == 2048


def test_sender_identity_uses_forwarder_and_anonymous_chat():
    forwarded = message(user=1)
    forwarded.forward_origin = SimpleNamespace(sender_user=SimpleNamespace(id=999))
    assert sender_key(forwarded) == (-100, "user", 1)
    forwarded.sender_chat = SimpleNamespace(id=-200)
    assert sender_key(forwarded) == (-100, "chat", -200)
