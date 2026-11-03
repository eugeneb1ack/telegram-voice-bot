# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot for voice messages.

- Transcribes voice messages with complete, expandable transcripts.
- Summarizes recordings.
- Combines multiple forwarded voice messages into one transcript and summary.

## Conversation context

A typing indicator runs during processing. Every third voice from the same author
(3, 6, 9, and so on) gets one combined reply: a summary of the current three voices
and the current recording’s full collapsible transcript. No separate summary message
is sent. The first two transcripts are delivered immediately. Nearby text from that author contributes context but does
not increment the voice counter. The sixth reply summarizes voices 4–6; earlier source messages clarify
references without being summarized again. A standalone recording of at least 30 seconds keeps its usual summary;
continuations are summarized on every third voice. Forwarded batches remain separate.

A gap over 180 seconds between the author's messages starts a new series. Defaults:
15-minute TTL, 12 messages and 12,000 characters per context, at most 200 contexts.
Older context is evicted when a limit is reached; Telegram transcripts stay complete.
Keys contain chat, forum topic and author IDs. Display names are not identifiers.
Forwarded voices use a separate scope. Private `/text` does not read or populate
public context; quiet mode does not collect text. Only received updates are used,
not historical Telegram messages.

Context and delivery markers survive restarts via an atomic snapshot with mode 0600.
One bot process must own one data directory; do not share it between instances.
Unprocessed voice jobs still queue in RAM. A storage failure disables cumulative
summaries while ordinary transcription continues. Summaries use the same queue and delivery deduplication as ordinary transcripts.
If the combined result exceeds Telegram’s limit, it is split while preserving the
entire transcript.

See `CONTEXT_*` in [.env.example](.env.example). `CONTEXT_SUMMARY_EVERY=0` disables
cumulative summaries; `CONTEXT_TTL_SECONDS=0` disables context collection entirely.
When upgrading, change an explicitly configured `CONTEXT_MAX_CHARS=800` to `12000`.

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

Temporary audio files are deleted after processing. Bounded excerpts of source
transcripts and ordinary text are stored in `conversations.json` in the `group-state`
Docker volume to survive restarts. Expired records are removed while the bot runs,
or on the next startup if it was stopped. This is not a chat archive. Other volume
files store group permissions and modes, usage counters, identifier hashes and public
Telegram reply IDs. Pending jobs and temporary delivery results remain in RAM.
Operational logs contain stage timings without audio, message text, API bodies,
credentials or Telegram exception contents. Audio, text and context are sent to
OpenAI for processing; replies remain in Telegram.

License: [MIT](LICENSE).
