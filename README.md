# Voice Bot

[English](README.md) | [Русский](README.ru.md)

Telegram bot for voice messages.

- Transcribes voice messages with complete, expandable transcripts.
- Summarizes recordings.
- Combines multiple forwarded voice messages into one transcript and summary.

## Recordings without recognized speech

Empty recordings and recognition failures produce no chat reply. They do not enter
conversation context or advance the three-voice counter. Forwarded batches include
only recognized speech, without failure notices for the remaining parts. An empty
result is durably completed: restart does not repeat STT or send a reply. Technical
failures are logged without speech content; an explicit retry remains possible after
the existing cooldown.

## Conversation context

A typing indicator runs during processing. Each new block of three voices by one
author receives one reply with that block's summary and the current recording's
full collapsible transcript. Both the third and sixth replies say “Коротко · 3
голосовых”; the latter covers voices 4–6. The author's text between those voices
also belongs to the block. Earlier text, other participants' messages and previous
aggregate summaries are not added automatically. A failed summary does not merge
its old block into the next one: complete transcripts are already delivered and
remain available as explicit reply sources. The first two short voices receive
transcripts only. A
recording of at least 30 seconds gets an immediate summary at any position in a
series. Pending shorter recordings are included with it, so they cannot disappear. After a
successful standalone summary, count three new voices. Earlier recordings provide
background through explicit replies or verified semantic continuation, using their
own transcript rather than an aggregate recap. A long first voice plus two short ones therefore produces no
second summary on message three: the new triple completes on message four. A failed
standalone summary does not mark the recording as summarized. Forwarded
batches remain separate. Round video notes count as voices and contribute speech
only. Ordinary video files are not transcribed; their captions can supply text.

A series spans at most 7 minutes from its first voice. The counter restarts at
that boundary: two voices followed by a third two hours later never form a triple.
Text does not extend the boundary. `CONTEXT_SERIES_GAP_SECONDS` (default 420) can
additionally limit silence between voices; `CONTEXT_TTL_SECONDS` caps total series
length.

Counters are per author, chat and actual forum topic. Background is selected by
explicit reply ancestry: the direct target plus at most two earlier ancestors
already available in memory. Neighbouring messages, sibling replies, the same
recipient and temporal proximity alone do not establish a connection. An explicit
old source is allowed without loading its surrounding history.

Without a reply, a separate bounded semantic selection precedes summary generation:
at most 12 candidates from the same chat/topic, 6,000 characters total. The search
window covers five minutes before each primary message, rather than before the final
summary. Source retention allows that lookback plus the seven-minute series (12 minutes
with defaults), under the same message/character caps. This does not extend counters.
Existing transcripts, text and captions may clarify a specific reference,
continuation, correction or answer; a broad shared topic is insufficient. Source IDs,
chronology and verbatim evidence quotes are validated. Ambiguous links are dropped.
The summary request sees at most three selected sources; rejected candidates never
reach it. Explicit replies take priority, and inferred continuity never invents a recipient.
A person reference retains the exact mention, source ID and source-backed name for
that primary message only. An inherited name is allowed only for a leading pronoun
in the first primary message, from that same author’s earlier text.
Multiple possible people, a topic switch or an unproven
name must not be resolved by guessing. An explicit reply already selected in the
current block can also clarify its continuation, regardless of the source's age.

Before generation, the application resolves each branch into separate current-speech
and background lists. It validates primary speakers by Telegram ID; recipient names
are presentation labels, never speaking-role instructions. Background retains its
own author and may clarify references without becoming new claims. The model no
longer joins a flat message graph. Answers use third-person narration, with
explicit “Author → Recipient” labels; unaddressed speech carries only the author's
name. Unknown people are explicitly left unspecified. Different recipients remain
separate. Display names take priority over usernames.
The summary must preserve negation, uncertainty and limits: listening to music does
not imply playing it, and one melody played with one finger does not imply general
proficiency. This adds no model call and does not alter original transcripts.
Semantic selection uses the configured summary model
only before a summary with available candidates. The first two short voices do not
wait for it. It has a four-second timeout; failure falls back to current messages and
explicit replies. Cached audio is never re-transcribed for context. Semantic selection
is probabilistic: validated evidence reduces risk but cannot guarantee perfect
interpretation of ambiguous speech.

Defaults: 12 messages and 12,000 characters per author, at most 200 active contexts.
Shared background has one additional 12,000-character budget and at most 100 nodes,
not a separate budget per participant. Durable counters survive source-text eviction.
If limits evict earlier sources, the third voice still gets a summary explicitly
marked as covering the available portion. Missing content is never reconstructed.
Telegram transcripts stay complete. Forwarded messages have separate scope; private
`/text` neither reads nor populates public memory. Quiet mode does not collect text.
Only received Telegram updates are used; old chat history is never fetched.

The first two short replies do not wait for an uncached older reply source to be
transcribed. Its bounded media metadata persists until the block summary and
survives restarts. Existing transcripts are reused. A summary can still take longer
when its older source has never been recognized.

Unprocessed voices and delivery progress persist in a SQLite journal. A context snapshot
failure disables block summaries; a queue journal failure stops polling before
acknowledging further updates. Summaries use the same queue and delivery deduplication as ordinary transcripts.
If the combined result exceeds Telegram’s limit, it is split while preserving the
entire transcript.

See `CONTEXT_*` in [.env.example](.env.example).
When upgrading, replace explicitly configured `CONTEXT_TTL_SECONDS=900` and
`CONTEXT_SERIES_GAP_SECONDS=900` with `420` for the seven-minute window. `CONTEXT_SUMMARY_EVERY=0` disables
block summaries; `CONTEXT_TTL_SECONDS=0` disables context collection entirely.
When upgrading, change an explicitly configured `CONTEXT_MAX_CHARS=800` to `12000`.

## Explicit reply context

When a voice replies to another message, summaries use that exact source as background:
text, captions (including bot messages), voice, or speech from a round video note.
Ordinary videos and other attachments contribute only their textual captions; the
attachment itself is never downloaded or transcribed for context. The source is attributed separately from
the author's own words. A reply on the first or second voice remains available for the
third voice's summary, including after restart. It does not increment the author's
voice counter and is not appended to the transcript. The next triple does not inherit
another person's source automatically: a fresh reply or verified semantic continuation
is required.

For explicit replies, the referenced message in the same chat is eligible. Other chats and ephemeral
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

Summary instructions distinguish events, intentions and attitudes: a reply alone does
not confirm a proposal, and reluctance does not establish whether an action happened.
Known references retain concrete details, while unrelated background is not repeated
as new content. The author's current texts may cover separate topics; microphone checks
and content-free counting are omitted from the summary, with full transcripts preserved.

## Spelling hints during transcription

`gpt-transcribe` receives up to 16 literal vocabulary hints (320 characters total)
from the explicit reply, the recording's caption and up to three preceding messages
from the same author, chat and forum topic within the 7-minute context window.
Existing voice transcripts may contribute vocabulary; source audio is never reprocessed
just to build hints. Typed spelling takes precedence over earlier speech recognition.
Only available text is used: an unrecognized referenced recording is not downloaded
for this step. Ordinary attachments contribute captions only.

Hints contain bounded individual words with capitalization, without usernames, URLs,
whole messages or conversational instructions. They use the dedicated `keywords[]`
field, not a transcript-rewriting pass. The returned transcript and whitespace stay
unchanged. A cached transcript is reused as-is; the hint list is checkpointed with a
pending job so a restart cannot change an in-flight request's context. Private `/text`
does not read ambient history. Disabling context disables these hints. Other configured
STT models keep their existing request format.

Hints can reduce spelling errors but cannot guarantee error-free recognition or the
absence of invented words. Only words actually spoken should appear; evaluate both
relevant names and unrelated/conflicting audio. The application adds the summary author label itself: the name comes
first, with `@username` only when no name is available. Forwarded messages retain their
original author.

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
Telegram reply IDs. Pending job metadata, Telegram file IDs, recognized text, summaries
and delivery progress are stored temporarily in `queue.sqlite3` (mode 0600). Audio is
not stored there. Completed job payloads are deleted; bounded request hashes prevent
replay. Unresolved jobs are retained until completed or explicitly reconciled.
Operational logs contain stage timings without audio, message text, API bodies,
credentials or Telegram exception contents. Audio, text and context are sent to
OpenAI for processing; replies remain in Telegram.

License: [MIT](LICENSE).
