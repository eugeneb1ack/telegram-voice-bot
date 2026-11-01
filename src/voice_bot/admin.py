import asyncio
import logging
import time
from html import escape

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import JOIN_TRANSITION, LEAVE_TRANSITION, ChatMemberUpdatedFilter, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    Chat,
    ChatMemberUpdated,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from voice_bot.config import Settings
from voice_bot.groups import Group, GroupStatus, GroupStore
from voice_bot.modes import settings_button

logger = logging.getLogger(__name__)
GREETING = (
    "Я перевожу голосовые в текст.\n\n"
    "Обычный режим — расшифровываю голосовые автоматически для всей группы.\n"
    "Тихий режим — только по /text в ответ на голосовое; результат видите только вы.\n\n"
    "Назначьте меня администратором. Выбрать режим: /settings."
)
PAGE_SIZE = 6
STATUS = {"pending": "⏳ Ожидает решения", "allowed": "✅ Разрешён", "denied": "⛔ Запрещён"}


class AdminAction(CallbackData, prefix="group"):
    action: str
    chat_id: int = 0
    revision: int = 0
    page: int = 0


def button(text: str, action: str, **kwargs: int) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=text, callback_data=AdminAction(action=action, **kwargs).pack()
    )


def group_card(group: Group) -> tuple[str, InlineKeyboardMarkup]:
    text = (
        f"<b>{escape(group.title)}</b>\n<code>{group.chat_id}</code>\n"
        f"{STATUS[group.status] if group.present else 'Бот удалён из группы'}"
        f"\nРежим: {'обычный' if group.mode == 'auto' else 'тихий'}"
    )
    rows = []
    if group.present:
        rows.append(
            [
                button("✅ Разрешить", "allow", chat_id=group.chat_id, revision=group.revision),
                button("⛔ Запретить", "deny", chat_id=group.chat_id, revision=group.revision),
            ]
        )
    rows.append([button("← Все группы", "list")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


class GroupAdmin:
    def __init__(self, settings: Settings, store: GroupStore) -> None:
        self.settings = settings
        self.store = store
        self.notification_attempts: dict[int, float] = {}
        self.notification_locks: dict[int, asyncio.Lock] = {}

    def allowed(self, message: Message) -> bool:
        if self.settings.allowed_chat_ids and message.chat.id not in self.settings.allowed_chat_ids:
            return False
        if message.chat.type == "private":
            return True
        return self.store.allowed(message.chat.id)

    async def observe(self, chat: Chat, bot: Bot, *, greet: bool = True) -> None:
        group, joined = self.store.observe(chat.id, chat.title or str(chat.id))
        # Try the private owner notification first; lack of group send rights must not block it.
        await self.notify(group, bot)
        if joined and greet:
            await self.safe_send(bot, chat.id, GREETING, settings_button())

    async def safe_send(
        self, bot: Bot, chat_id: int, text: str, markup: InlineKeyboardMarkup | None = None
    ) -> bool:
        try:
            await bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=markup)
            return True
        except TelegramAPIError:
            logger.warning("admin_delivery_failed chat_id=%s", chat_id)
            return False

    async def edit_card(self, message: Message, text: str, markup: InlineKeyboardMarkup) -> None:
        if message.html_text == text and message.reply_markup == markup:
            return
        try:
            await message.edit_text(text, parse_mode="HTML", reply_markup=markup)
        except TelegramBadRequest as error:
            if "message is not modified" not in error.message.casefold():
                raise

    async def notify(self, group: Group, bot: Bot) -> None:
        lock = self.notification_locks.setdefault(group.chat_id, asyncio.Lock())
        async with lock:
            # A concurrent /start or join may already have delivered this request.
            current = self.store.groups.get(group.chat_id)
            if current is not None:
                await self._notify(current, bot)

    async def _notify(self, group: Group, bot: Bot) -> None:
        if group.status != "pending" or group.notified or not group.present:
            return
        previous = self.notification_attempts.get(group.chat_id, float("-inf"))
        if time.monotonic() - previous < 30:
            return
        self.notification_attempts[group.chat_id] = time.monotonic()
        text, markup = group_card(group)
        if await self.safe_send(
            bot, self.settings.owner_id, "Бота добавили в группу.\n\n" + text, markup
        ):
            self.store.mark_notified(group)
            logger.info("group_approval_requested chat_id=%s", group.chat_id)

    async def notify_pending(self, bot: Bot) -> None:
        for group in list(self.store.groups.values()):
            await self.notify(group, bot)

    def panel(self, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
        groups = sorted(
            self.store.groups.values(),
            key=lambda group: (
                group.status != "pending",
                not group.present,
                group.title.casefold(),
            ),
        )
        pages = max(1, (len(groups) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(max(page, 0), pages - 1)
        pending = sum(g.status == "pending" and g.present for g in groups)
        allowed = sum(g.status == "allowed" and g.present for g in groups)
        text = (
            f"<b>Управление группами</b>\n"
            f"Ожидают: {pending} · Разрешены: {allowed} · Всего: {len(groups)}\n"
            "\n"
            "Выберите группу, чтобы разрешить или запретить обработку."
        )
        rows = [
            [
                button(
                    f"{STATUS[g.status].split()[0] if g.present else '➖'} {g.title[:45]}",
                    "view",
                    chat_id=g.chat_id,
                )
            ]
            for g in groups[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
        ]
        if pages > 1:
            rows.append(
                [
                    button("←", "list", page=max(0, page - 1)),
                    button(f"{page + 1}/{pages}", "list", page=page),
                    button("→", "list", page=min(pages - 1, page + 1)),
                ]
            )
        rows.append([button("Обновить", "list", page=page)])
        return text, InlineKeyboardMarkup(inline_keyboard=rows)

    async def callback(self, query: CallbackQuery, callback_data: AdminAction, bot: Bot) -> None:
        if (
            query.from_user.id != self.settings.owner_id
            or not isinstance(query.message, Message)
            or query.message.chat.type != "private"
            or query.message.chat.id != self.settings.owner_id
        ):
            await query.answer("Только для владельца бота.", show_alert=True)
            return
        action = callback_data.action
        if action in {"allow", "deny"}:
            desired: GroupStatus = "allowed" if action == "allow" else "denied"
            current = self.store.groups.get(callback_data.chat_id)
            if current is not None and current.present and current.status == desired:
                await query.answer("Уже разрешено" if action == "allow" else "Уже запрещено")
                return
            group = self.store.decide(
                callback_data.chat_id,
                callback_data.revision,
                desired,
            )
            if group is None:
                await query.answer("Кнопка устарела. Откройте группу заново.", show_alert=True)
                return
            logger.info("group_decision chat_id=%s status=%s", group.chat_id, group.status)
            await query.answer("Разрешено" if action == "allow" else "Запрещено")
            text, markup = group_card(group)
            await self.edit_card(query.message, text, markup)
            await self.safe_send(
                bot,
                group.chat_id,
                (
                    "Обработка голосовых разрешена."
                    if action == "allow"
                    else "Обработка голосовых отключена владельцем."
                ),
            )
        elif action == "view":
            group = self.store.groups.get(callback_data.chat_id)
            if group is None:
                await query.answer("Группа не найдена.", show_alert=True)
                return
            await query.answer()
            text, markup = group_card(group)
            await self.edit_card(query.message, text, markup)
        elif action == "list":
            await query.answer()
            text, markup = self.panel(callback_data.page)
            await self.edit_card(query.message, text, markup)
        else:
            await query.answer("Неизвестная команда.", show_alert=True)


def build_admin_router(admin: GroupAdmin) -> Router:
    router = Router(name="admin")

    @router.my_chat_member(ChatMemberUpdatedFilter(JOIN_TRANSITION))
    async def joined(event: ChatMemberUpdated, bot: Bot) -> None:
        if event.chat.type in {"group", "supergroup"}:
            await admin.observe(event.chat, bot)

    @router.my_chat_member(ChatMemberUpdatedFilter(LEAVE_TRANSITION))
    async def left(event: ChatMemberUpdated) -> None:
        admin.store.leave(event.chat.id)
        logger.info("group_left chat_id=%s", event.chat.id)

    @router.message(Command("start", "help"))
    async def start(message: Message, bot: Bot) -> None:
        if message.chat.type in {"group", "supergroup"}:
            await admin.observe(message.chat, bot, greet=False)
            await message.answer(GREETING, reply_markup=settings_button())
            return
        me = await bot.me()
        rows = [
            [
                InlineKeyboardButton(
                    text="Добавить в группу",
                    url=f"https://t.me/{me.username}?startgroup=true",
                )
            ]
        ]
        if message.from_user and message.from_user.id == admin.settings.owner_id:
            rows.append([button("Управление группами", "list")])
        await message.answer(GREETING, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
        if message.from_user and message.from_user.id == admin.settings.owner_id:
            admin.notification_attempts.clear()
            await admin.notify_pending(bot)

    @router.message(Command("admin"), F.chat.type == "private")
    async def panel(message: Message) -> None:
        if not message.from_user or message.from_user.id != admin.settings.owner_id:
            return
        text, markup = admin.panel()
        await message.answer(text, parse_mode="HTML", reply_markup=markup)

    router.callback_query.register(admin.callback, AdminAction.filter())
    return router
