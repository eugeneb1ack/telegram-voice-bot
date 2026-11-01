# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram-бот для голосовых сообщений.

- Расшифровывает голосовые сообщения.
- Составляет краткий пересказ записей.
- Объединяет несколько пересланных голосовых в одну расшифровку и общий пересказ.

## Развёртывание

Нужны Linux, Git, Docker Engine и Docker Compose v2. Входящие порты не используются.

```bash
git clone https://github.com/eugeneb1ack/telegram-voice-bot.git
cd telegram-voice-bot
test -e .env || cp .env.example .env
chmod 600 .env
```

Заполните в `.env`:

| Переменная | Значение |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Токен от BotFather |
| `OPENAI_API_KEY` | Ключ OpenAI API |
| `OWNER_ID` | Telegram ID владельца |

Остальные настройки описаны в [.env.example](.env.example).

```bash
docker compose build
docker compose run --rm --no-deps voice-bot voice-bot --check
docker compose up -d
docker compose ps
```

Добавьте бота в группу и назначьте администратором. Затем с аккаунта, указанного
в `OWNER_ID`, отправьте боту `/admin` в личном чате и разрешите обработку для группы.

## Данные

Временные аудиофайлы удаляются после обработки. Расшифровки не записываются на диск
и не попадают в служебные логи. Очередь, контекст и временные результаты доставки
хранятся в RAM и автоматически очищаются. Docker volume `group-state` хранит разрешения
групп, счётчики использования и хеши идентификаторов обработанных записей.
Служебные логи не содержат текстов голосовых.
Аудио, текст и контекст передаются в OpenAI для обработки; ответы остаются в Telegram.

Лицензия — [MIT](LICENSE).
