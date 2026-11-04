# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot that transcribes voice messages and round video notes using OpenAI.

- Full transcripts in expandable quotes, with a typing indicator during processing.
- Summaries for recordings of at least 15 seconds, or each new group of three shorter recordings within 10 minutes.
- Reply context from transcripts, text and captions; separate counters per user, chat and topic.
- Automatic processing or quiet mode with private `/text` replies.
- Persistent queue for restart recovery. No reply when speech is not recognized.

## Deployment

Requires Docker with Compose and an OpenAI API key with access to the configured models.

```sh
git clone https://github.com/eugeneb1ack/telegram-voice-bot.git
cd telegram-voice-bot
test -e .env || cp .env.example .env
```

Set these values in `.env`:

| Variable | Value |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather |
| `OPENAI_API_KEY` | OpenAI API key |
| `OWNER_ID` | Bot owner's Telegram user ID |

Models, limits and proxy settings are listed in [.env.example](.env.example).

```sh
docker compose up -d --build
docker compose logs -f voice-bot
```

For groups, make the bot an administrator, then approve the group through `/admin` in a private chat with the bot.

## Commands

| Command | Action |
| --- | --- |
| `/text` | Transcribe the recording being replied to; only the requester sees the result in groups |
| `/settings` | Switch group mode: automatic or quiet; group administrators only |
| `/start` | Show help |
| `/admin` | Approve or block groups; owner only |

State is stored in the Docker volume. Audio and selected text context are sent to OpenAI for processing.

[MIT License](LICENSE)
