# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot for voice messages.

- Transcribes voice messages.
- Summarizes recordings.
- Combines multiple forwarded voice messages into one transcript and summary.

## Deployment

Requires Linux, Git, Docker Engine and Docker Compose v2. No inbound ports are needed.

```bash
git clone https://github.com/eugeneb1ack/telegram-voice-bot.git
cd telegram-voice-bot
test -e .env || cp .env.example .env
chmod 600 .env
```

Set these values in `.env`:

| Variable | Value |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | BotFather token |
| `OPENAI_API_KEY` | OpenAI API key |
| `OWNER_ID` | Owner's Telegram ID |

Other settings are listed in [.env.example](.env.example).

```bash
docker compose build
docker compose run --rm --no-deps voice-bot voice-bot --check
docker compose up -d
docker compose ps
```

Add the bot to a group and make it an administrator. From the Telegram account
specified in `OWNER_ID`, send `/admin` to the bot in a private chat and approve the group.

## Data

Temporary audio files are deleted after processing. Transcripts are not saved to disk
or included in application logs. Pending jobs and recent context stay in RAM;
context and temporary delivery results are automatically cleared. The Docker volume
`group-state` stores group permissions, usage counters and hashes of processed message identifiers.
Operational logs contain no voice message content.
Audio, text and context are sent to OpenAI for processing; replies remain in Telegram.

License: [MIT](LICENSE).
