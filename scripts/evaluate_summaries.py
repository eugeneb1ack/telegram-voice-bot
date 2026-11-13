"""Run the versioned synthetic summary corpus; API calls require --execute.

The report contains model output and a human review rubric. It deliberately does
not label semantic quality as passing based on a keyword match or another model.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from voice_bot.config import Settings
from voice_bot.dialogue_graph import DialogueBranch, DialogueMessage, SharedDialogue
from voice_bot.dialogue_summary import has_summary_content
from voice_bot.providers import APIClient, HTTPSummarizer, SummaryContext, VoicePartContext
from voice_bot.reply_context import ReplyContext


def source_context(case: dict[str, Any]) -> SummaryContext:
    sources = case["sources"]
    first = sources[0]["speaker"]
    same_author = all(source["speaker"]["id"] == first["id"] for source in sources)
    references = tuple(
        ReplyContext(
            message_id=source["message_id"],
            reply_to_message_id=-index,
            author_name=source["reply_recipient"]["name"],
            author_username=None,
            author_id=source["reply_recipient"]["id"],
            kind="recipient",
            text="",
        )
        for index, source in enumerate(sources, 1)
        if source.get("reply_recipient")
    )
    context = SummaryContext(
        author_name=first["name"] if same_author else None,
        author_source="user" if same_author else "multiple",
        reply_contexts=references,
    )
    if case["mode"] == "batch":
        return SummaryContext(
            author_name=context.author_name,
            author_source=context.author_source,
            reply_contexts=references,
            voices=tuple(
                VoicePartContext(
                    index=index,
                    author_name=source["speaker"]["name"],
                    author_username=None,
                    duration_seconds=None,
                    message_id=source["message_id"],
                    author_id=source["speaker"]["id"],
                    text=source["text"],
                )
                for index, source in enumerate(sources, 1)
            ),
        )
    if case["mode"] != "dialogue":
        return context
    if not same_author:
        raise ValueError("A dialogue evaluation must contain one primary author")
    refs = {ref.message_id: ref for ref in references}
    nodes = [
        DialogueMessage(
            source["message_id"],
            source["speaker"]["id"],
            source["speaker"]["name"],
            None,
            "voice",
            source["text"],
            refs[source["message_id"]].reply_to_message_id
            if source["message_id"] in refs
            else None,
        )
        for source in sources
    ]
    nodes.extend(
        DialogueMessage(
            ref.reply_to_message_id, ref.author_id, ref.author_name, None, "recipient", ""
        )
        for ref in references
    )
    return SummaryContext(
        author_name=context.author_name,
        author_source=context.author_source,
        is_conversation=True,
        dialogue=SharedDialogue(
            focus_author_id=first["id"],
            messages=tuple(nodes),
            branches=(
                DialogueBranch(
                    "evaluation", None, None, None, tuple(s["message_id"] for s in sources)
                ),
            ),
        ),
    )


async def evaluate(cases: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    results = []
    async with httpx.AsyncClient(
        proxy=settings.openai_proxy_url.get_secret_value() or None,
        timeout=settings.request_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        summarizer = HTTPSummarizer(APIClient(client, settings))
        for case in cases:
            started = time.monotonic()
            should_summarize = any(has_summary_content(s["text"]) for s in case["sources"])
            result: dict[str, Any] = {
                "id": case["id"],
                "expected": case["expected"],
                "should_summarize": should_summarize,
            }
            try:
                if should_summarize != case["expected"]["should_summarize"]:
                    raise ValueError("Unexpected summary eligibility")
                if should_summarize:
                    async with asyncio.timeout(settings.job_timeout_seconds):
                        result["summary"] = await summarizer.summarize(
                            "\n\n".join(source["text"] for source in case["sources"]),
                            context=source_context(case),
                        )
                    result["status"] = "review_required"
                else:
                    result["status"] = "skipped_without_api"
            except Exception as error:
                result.update(status="error", error_type=type(error).__name__)
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/summary-eval-v1.json",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--execute", action="store_true", help="Make paid summary API requests")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    cases = corpus["cases"]
    if args.case_ids:
        requested = set(args.case_ids)
        if not requested <= {case["id"] for case in cases}:
            parser.error("Unknown evaluation case")
        cases = [case for case in cases if case["id"] in requested]
    if corpus["version"] != 1 or not 1 <= len(cases) <= 8:
        parser.error("Expected corpus version 1 and between 1 and 8 cases")
    if not args.execute:
        print(json.dumps({"dry_run": True, "cases": [case["id"] for case in cases]}))
        return 0
    settings = Settings()
    if not settings.openai_api_key.get_secret_value().strip():
        parser.error("OPENAI_API_KEY is required")
    results = asyncio.run(evaluate(cases, settings))
    print(
        json.dumps(
            {
                "corpus_version": corpus["version"],
                "model": settings.summary_model,
                "reasoning": settings.summary_reasoning_effort,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(any(result["status"] == "error" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
