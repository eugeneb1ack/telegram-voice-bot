"""Offline recovery of ambiguous Telegram sends; never contacts Telegram."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from voice_bot.journal import QueueJournal


def status(path: Path) -> list[dict[str, Any]]:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT id,state,payload,deliveries FROM jobs ORDER BY created"
        ).fetchall()
        return [
            {
                "job_id": job_id,
                "state": state,
                "chat_id": json.loads(payload)[-1]["message"]["chat"]["id"],
                "voice_ids": [item["message"]["message_id"] for item in json.loads(payload)],
                "confirmed_parts": sum(
                    p["state"] == "sent" for p in json.loads(deliveries).values()
                ),
                "uncertain_parts": sum(
                    p["state"] == "sending" for p in json.loads(deliveries).values()
                ),
            }
            for job_id, state, payload, deliveries in rows
        ]
    finally:
        db.close()


def resolve(path: Path, job_id: str, message_id: int | None) -> None:
    # The exclusive owner lock requires stopping the bot before repair.
    journal = QueueJournal(path)
    try:
        job = journal.get(job_id)
        unknown = [key for key, part in job["deliveries"].items() if part["state"] == "sending"]
        if job["state"] != "uncertain" or len(unknown) != 1:
            raise ValueError("Job must have exactly one uncertain send")
        if message_id is None:
            journal.rejected(job_id, unknown[0])
        else:
            if message_id < 0:
                raise ValueError("Invalid message ID")
            journal.sent(job_id, unknown[0], message_id)
        journal.set(job_id, "state", "queued")
    finally:
        journal.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка и восстановление очереди Voice Bot")
    parser.add_argument("--path", type=Path, default=Path("data/queue.sqlite3"))
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("status", help="Только ID и состояния; без текстов и аудио")
    repair = commands.add_parser("resolve", help="Только после проверки Telegram и остановки бота")
    repair.add_argument("job_id")
    decision = repair.add_mutually_exclusive_group(required=True)
    decision.add_argument(
        "--delivered-id", type=int, help="Подтверждённый ID ответа; 0 для ephemeral"
    )
    decision.add_argument(
        "--not-delivered", action="store_true", help="Проверено: ответ не отправлен"
    )
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(json.dumps(status(args.path), ensure_ascii=False, indent=2))
        else:
            resolve(args.path, args.job_id, args.delivered_id)
            print("Состояние исправлено. После запуска бот продолжит задание.")
    except Exception as exc:
        parser.exit(1, f"Операция не выполнена ({type(exc).__name__}). Значения скрыты.\n")


if __name__ == "__main__":
    main()
