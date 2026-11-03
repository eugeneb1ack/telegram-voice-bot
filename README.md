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
Keys contain chat, actual forum topic and author IDs. Ordinary replies stay in the
same author's series; a reply-chain ID is not a forum topic. Display names are not identifiers.
Forwarded voices use a separate scope. Private `/text` does not read or populate
public context; quiet mode does not collect text. Only received updates are used,
not historical Telegram messages.

Context and delivery markers survive restarts via an atomic snapshot with mode 0600.
One bot process must own one data directory; do not share it between instances.
Unprocessed voices and delivery progress persist in a SQLite journal. A context snapshot
failure disables cumulative summaries; a queue journal failure stops polling before
acknowledging further updates. Summaries use the same queue and delivery deduplication as ordinary transcripts.
If the combined result exceeds Telegram’s limit, it is split while preserving the
entire transcript.

See `CONTEXT_*` in [.env.example](.env.example). `CONTEXT_SUMMARY_EVERY=0` disables
cumulative summaries; `CONTEXT_TTL_SECONDS=0` disables context collection entirely.
When upgrading, change an explicitly configured `CONTEXT_MAX_CHARS=800` to `12000`.

## Explicit reply context

When a voice replies to another message, summaries use that exact source as background:
text, captions (including bot messages), voice, or speech from a round video note.
Ordinary videos and other attachments contribute only their textual captions; the
attachment itself is never downloaded or transcribed for context. The source is attributed separately from
the author's own words. A reply on the first or second voice remains available for the
third voice's summary, including after restart. It does not increment the author's
voice counter and is not appended to the transcript. The next triple does not inherit
another person's source unless one of its messages explicitly replies to it again.

Only the referenced message in the same chat is eligible. Other chats and ephemeral
sources are excluded; no surrounding history of the other participant is fetched.
An explicit reference across forum topics may provide background, while each topic's
voice counter remains separate. Ordinary reply chains are not forum topics.

Existing source transcripts are reused. If speech from the referenced voice/video note has
not been recognized yet, it uses the same bounded download, FFmpeg and STT path inside
the current job. Additional context work has a 45-second ceiling and reserves time
for the summary. Video notes contribute speech only, not visual interpretation. Unavailable
or oversized sources do not prevent delivery of the author's transcript; the summary
must not invent missing context. No models have changed.

Referenced text is bounded to 3,000 characters per voice and 6,000 per summary, also
subject to the existing conversation character budget. The bounded source and explicit
reply relation are persisted with the conversation; pending jobs store one shallow
source snapshot, not a recursive reply history. Summary prompts distinguish the source
question/proposal, the author's response/clarification and the actual outcome, without
adding decisions, unnecessary sections or filler.

## Durable queue

Accepted voices, including forwarded voices still being collected, survive process
restarts. Queue admission commits before polling acknowledges the update. Scheduling
remains bounded and ordered within each chat. Recognized text and prepared summaries
are checkpointed, so recovery reuses completed model results. Previous voices used as
summary context are never transcribed again.

Every outgoing part is journaled before sending and its Telegram ID is committed
immediately after confirmation. Recovery skips confirmed parts, including long replies
and cached copies. A new explicit `/text` command remains a new private delivery.
The same polling update is not a new request. Existing queue/concurrency limits apply.
The journal uses standard-library SQLite, WAL and FULL synchronization, and an exclusive
process lock; no new service or dependency is needed. `QUEUE_STATE_FILE` defaults to
`data/queue.sqlite3` and Compose places it in the persistent data volume.

**Transport limit:** Telegram Bot API has no send idempotency key. A crash or network
loss between sending and saving the acknowledgment leaves delivery uncertain. Such a
job is retained and is **not automatically resent**. This avoids duplicates, but cannot
guarantee both zero loss and exactly-once delivery in that window. Check the chat before
resolving it. This also applies to a process killed just before its socket write.

Inspect metadata without texts/audio while the bot is running:

```bash
docker compose exec voice-bot python -m voice_bot.queue_admin --path /app/data/queue.sqlite3 status
```

For an uncertain job, verify the exact reply in Telegram, stop the bot, then run one of:

```bash
docker compose stop voice-bot
# Confirmed reply ID (0 only for a verified ephemeral reply):
docker compose run --rm --no-deps voice-bot python -m voice_bot.queue_admin --path /app/data/queue.sqlite3 resolve JOB_ID --delivered-id REPLY_ID
# OR, only after verifying that the reply was not delivered:
docker compose run --rm --no-deps voice-bot python -m voice_bot.queue_admin --path /app/data/queue.sqlite3 resolve JOB_ID --not-delivered
docker compose up -d --no-build
```

Do not delete the journal to clear a backlog. Back up the entire data volume while the
bot is stopped. Upgrading from 0.3.0 should use a graceful stop to drain its RAM queue;
voices accepted by older code cannot be recovered retroactively.

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
git clone https://github.com/eugeneb1ack/voice-bot.git
cd voice-bot
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
Telegram reply IDs. Pending job metadata, Telegram file IDs, recognized text, summaries
and delivery progress are stored temporarily in `queue.sqlite3` (mode 0600). Audio is
not stored there. Completed job payloads are deleted; bounded request hashes prevent
replay. Unresolved jobs are retained until completed or explicitly reconciled.
Operational logs contain stage timings without audio, message text, API bodies,
credentials or Telegram exception contents. Audio, text and context are sent to
OpenAI for processing; replies remain in Telegram.

License: [MIT](LICENSE).
