# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot that transcribes voice messages using OpenAI.

- Full transcripts in expandable quotes, with a typing indicator during processing.
- Summaries for recordings of at least 15 seconds, or each group of three shorter recordings with its own 7-minute window.
- Summaries use only the current voice recordings; separate counters per user, chat and topic.
- Text, captions, video notes and earlier voice blocks are excluded. Reply recipient names are preserved.
- Automatic processing or quiet mode with private `/text` replies.
- Persistent queue for restart recovery. No reply when speech is not recognized.
- Greetings and filler-only recordings keep their full transcript without an unnecessary summary.

## Getting started

Send or forward a voice message to the bot. Expand the quote to read the full transcript.
Use `/start` or `/help` for a short guide. To connect a group, tap **Добавить в группу**,
make the bot an administrator, and wait for the bot owner to approve the automatic request.
In a group, `/help` shows its current access and processing mode.

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
| `/start` | Start using the bot and add it to a group |
| `/help` | Short guide; current access and mode in groups |
| `/admin` | Approve or block groups; owner only |

State is stored in the Docker volume. Audio and its transcripts are sent to OpenAI for processing.

[MIT License](LICENSE)

### Summary fidelity and cost

Summaries preserve the original tense, uncertainty, requests, and corrections.
`SUMMARY_REASONING_EFFORT=low` enables lightweight Luna reasoning; `none` is available
for minimum cost. Existing summary length settings and the 600-second voice limit
are unchanged. Reasoning shares `SUMMARY_MAX_OUTPUT_TOKENS` with visible text. Only
an incomplete response caused by that token budget retries once with `none`; partial
summaries are never published. STT/transcript fallback remains available.

### Reliable delivery and verification

The container uses Python 3.14.10; CI also checks Python 3.12.16. Dependencies remain
locked. After Telegram explicitly rejects a prepared reply with `RetryAfter`, the
durable queue retries up to three times within `JOB_TIMEOUT_SECONDS`, preserving
chat order and already delivered parts. Waiting does not occupy a worker or repeat
STT/summary calls. Ambiguous network outcomes remain blocked for review to avoid
duplicate sends. A blocked bot or a change to quiet mode stops delivery.

Every summary uses a strict structured response contract. Only complete assistant
answers are accepted; refused or incomplete text is never salvaged. Each recognized
voice is paired with its own speaker and recipient, including partial batches.

The synthetic semantic corpus covers tense, negation, uncertainty, corrections,
recipients, instruction injection, multiple speakers and filler-only speech:

```sh
uv run python scripts/evaluate_summaries.py
# Paid API evaluation; requires OPENAI_API_KEY:
uv run python scripts/evaluate_summaries.py --execute
```

The default command makes no API calls. The evaluation report includes the output
and a human review rubric; a valid JSON response alone is not a semantic pass.
