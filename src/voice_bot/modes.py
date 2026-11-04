"""Group administrators choose persistent processing modes using private buttons."""

from typing import TYPE_CHECKING

from aiogram import Bot, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    EphemeralMessageParameters,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyParameters,
)

from voice_bot.delivery import personal, reply_to
from voice_bot.groups import Group

if TYPE_CHECKING:
    from voice_bot.admin import GroupAdmin


class ModeAction(CallbackData, prefix="mode"):
    action: str = "open"
    revision: int = 0


def settings_button() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚙️ Режим работы", callback_data=ModeAction().pack())]
        ]
    )


def mode_panel(group: Group) -> tuple[str, InlineKeyboardMarkup]:
    mode = "Обычный" if group.mode == "auto" else "Тихий"
    text = (
        f"<b>Режим работы: {mode}</b>\n\n"
        "<b>Обычный</b> — автоматически расшифровываю голосовые для всей группы.\n"
        "<b>Тихий</b> — обрабатываю только /text в ответ на голосовое.\n\n"
        "Ответ на /text в обоих режимах видите только вы. "
        "Для скрытой команды выберите /text в меню Telegram. "
        "Бот должен быть администратором.\n"
        "Менять режим могут администраторы группы."
    )
    if group.status != "allowed":
        text += "\n\nОбработка начнётся после разрешения владельца бота."
    return text, InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=("✓ " if group.mode == value else "") + label,
                    callback_data=ModeAction(action=value, revision=group.revision).pack(),
                )
                for value, label in (("auto", "Обычный"), ("quiet", "Тихий"))
            ]
        ]
    )


async def is_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in {"administrator", "creator"}
    except TelegramAPIError:
        return False


async def refresh_panel(query: CallbackQuery, bot: Bot, group: Group, *, opening: bool) -> None:
    assert isinstance(query.message, Message)
    text, markup = mode_panel(group)
    if not opening:
        try:
            if query.message.ephemeral_message_id:
                await bot.edit_ephemeral_message_text(
                    group.chat_id,
                    query.from_user.id,
                    query.message.ephemeral_message_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            else:
                await bot.edit_message_text(
                    text,
                    chat_id=group.chat_id,
                    message_id=query.message.message_id,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            return
        except TelegramBadRequest as exc:
            if "message is not modified" in exc.message.casefold():
                return
            # Expired or deleted menus can be replaced with a fresh private panel.
    await bot.send_message(
        group.chat_id,
        text,
        parse_mode="HTML",
        reply_markup=markup,
        ephemeral_message_parameters=EphemeralMessageParameters(
            receiver_user_id=query.from_user.id
        ),
    )


def build_modes_router(admin: "GroupAdmin") -> Router:
    router = Router(name="modes")

    @router.message(Command("settings"), flags={"cleanup_command": True})
    async def settings(message: Message, bot: Bot) -> None:
        if message.chat.type not in {"group", "supergroup"}:
            await message.answer("Откройте /settings в группе, в которой нужно выбрать режим.")
            return
        target = personal(message)
        if target is None or target.receiver_user is None:
            return
        await admin.observe(message.chat, bot, greet=False)
        parameters = (
            ReplyParameters(ephemeral_message_id=message.ephemeral_message_id)
            if message.ephemeral_message_id
            else reply_to(target)
        )
        if not await is_admin(bot, message.chat.id, target.receiver_user.id):
            text, markup = "Менять режим могут только администраторы группы.", None
        elif not await is_admin(bot, message.chat.id, bot.id):
            text, markup = "Назначьте бота администратором и снова откройте /settings.", None
        else:
            text, markup = mode_panel(admin.store.groups[message.chat.id])
        await bot.send_message(
            message.chat.id,
            text,
            parse_mode="HTML",
            reply_markup=markup,
            reply_parameters=parameters,
            ephemeral_message_parameters=EphemeralMessageParameters(
                receiver_user_id=target.receiver_user.id
            ),
        )

    @router.callback_query(ModeAction.filter())
    async def choose(query: CallbackQuery, callback_data: ModeAction, bot: Bot) -> None:
        if not isinstance(query.message, Message) or query.message.chat.type not in {
            "group",
            "supergroup",
        }:
            await query.answer("Откройте /settings в группе.", show_alert=True)
            return
        chat_id = query.message.chat.id
        if not await is_admin(bot, chat_id, query.from_user.id):
            await query.answer("Менять режим могут только администраторы группы.", show_alert=True)
            return
        if not await is_admin(bot, chat_id, bot.id):
            await query.answer("Сначала назначьте бота администратором.", show_alert=True)
            return
        await admin.observe(query.message.chat, bot, greet=False)
        group = admin.store.groups[chat_id]
        action = callback_data.action
        stale = False
        if action in {"auto", "quiet"}:
            updated = admin.store.set_mode(
                chat_id, callback_data.revision, "auto" if action == "auto" else "quiet"
            )
            if updated is None:
                stale = True
            else:
                group = updated
        elif action != "open":
            await query.answer("Неизвестный режим.", show_alert=True)
            return
        await refresh_panel(query, bot, group, opening=action == "open")
        await query.answer(
            "Меню обновлено. Выберите режим ещё раз."
            if stale
            else f"Режим: {'обычный' if group.mode == 'auto' else 'тихий'}"
        )

    return router
