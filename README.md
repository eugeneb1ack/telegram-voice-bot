# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot for voice messages.

- Transcribes voice messages with complete, expandable transcripts.
- Summarizes recordings.
- Combines multiple forwarded voice messages into one transcript and summary.

## Group modes

| Mode | Behavior |
| --- | --- |
| Normal (default) | Automatically transcribes voice messages for the whole group. |
| Quiet | Processes voice messages only on `/text`; only the requester sees the result. |

Group administrators select a mode with buttons in `/settings` or the welcome message.
The setting is stored per group and survives restarts. Existing groups keep normal mode.
The bot owner's approval is required in either mode.

## Commands

| Command | Action |
| --- | --- |
| `/text` | Transcribe the voice message you reply to; group replies are private in either mode. |
| `/settings` | Select the group's mode; group administrators only. |
| `/start` | Show help and group setup buttons. |
| `/admin` | Approve or block groups; bot owner only, in a private chat. |

Group `/text` replies are native Telegram **ephemeral messages**, including errors and
all parts of a long transcript. If private delivery fails, the bot does not publish the result.
`/text` and `/settings` are registered as ephemeral group commands.
To hide the command itself, reply to a voice message and select the private `/text`
from Telegram's command menu in a supporting client. If several bots are present,
select this bot's command (`/text@bot_username`).

Public `/text` and `/settings` messages from older clients or userbots are deleted after
five seconds if the bot can delete messages. The original voice message is kept.
Commands in private chats are not deleted; automatic processing and daily limits still apply.

The bot must be a group administrator. Ephemeral messages are temporary and may disappear
when Telegram restarts; offline delivery is not guaranteed. `/text` and `/settings`
sent anonymously on behalf of the group are not processed. See [Telegram's ephemeral message documentation](https://core.telegram.org/bots/api#ephemeral-messages-and-commands).

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
`group-state` stores group permissions and modes, usage counters, identifier hashes and
public Telegram reply IDs.
Operational logs contain no voice message content.
Audio, text and context are sent to OpenAI for processing; replies remain in Telegram.

License: [MIT](LICENSE).
