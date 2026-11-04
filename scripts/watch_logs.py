"""Читаемый монитор существующего потока логов; не подключается к API бота."""

import argparse
import re
import sys
import time
from collections import deque
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

LINE = re.compile(
    r"^(?:\S+Z )?(?P<date>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ "
    r"(?P<level>\w+) (?P<logger>\S+) (?P<event>\S+)(?P<fields>.*)$"
)
LABELS = {
    "voice_job_queued": ("В очереди", "yellow"),
    "voice_queue_wait": ("Ожидание", "dim"),
    "voice_job_started": ("Обработка", "cyan"),
    "typing_started": ("Печатает…", "dim"),
    "audio_downloaded": ("Аудио загружено", "cyan"),
    "audio_converted": ("Аудио готово", "cyan"),
    "transcription_completed": ("Речь распознана", "bright_cyan"),
    "context_selection_completed": ("Контекст выбран", "magenta"),
    "context_selection_skipped": ("Без доп. контекста", "dim"),
    "summary_completed": ("Итог готов", "bright_magenta"),
    "voice_job_delivered": ("Ответ отправлен", "green"),
    "voice_job_finished": ("Завершено", "bold green"),
    "summary_dialogue_prepared": ("Связи реплик", "magenta"),
    "provider_request_completed": ("Запрос API", "blue"),
}
FIELDS = {
    "pending": "в очереди",
    "parts": "частей",
    "sources": "источников",
    "primary": "реплик",
    "replies": "реплаев",
    "linked": "связей",
    "background": "фон",
    "attempt": "попытка",
    "chat_id": "чат",
    "topic_id": "тема",
}


def parse_line(line: str, zone: ZoneInfo) -> tuple[str, str, str, str, str, str]:
    match = LINE.match(line.strip())
    if not match:
        # Never interpret log text as Rich markup or terminal escape sequences.
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line.strip())
        clean = "".join(c for c in clean if c.isprintable())
        return "—", "dim", "Сервис", "—", "", clean[:500]
    stamp, level, logger, event, raw = match.groups()
    clock = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    fields = dict(re.findall(r"(\w+)=([^\s]+)", raw))
    label, style = LABELS.get(event, (event, "white"))
    if level in {"WARNING", "ERROR", "CRITICAL"}:
        label = f"{level} · {label}"
        style = "bold red" if level != "WARNING" else "bold yellow"
    elapsed = fields.pop("elapsed_ms", "")
    duration = f"{int(elapsed) / 1000:.2f} с" if elapsed.isdigit() else elapsed
    mid = fields.pop("message_id", "—")
    details = " · ".join(f"{FIELDS.get(k, k)}: {v}" for k, v in fields.items())
    if not details and event not in LABELS:
        details = raw.strip()[:500] or logger.removeprefix("voice_bot.")
    return clock.astimezone(zone).strftime("%H:%M:%S"), style, label, mid, duration, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Voice Bot · production")
    parser.add_argument("--timezone", default="Europe/Moscow")
    args = parser.parse_args()
    zone = ZoneInfo(args.timezone)
    console = Console(highlight=False)
    rows: deque[tuple[str, str, str, str, str, str]] = deque(maxlen=100)
    started = time.monotonic()
    last = started
    finished = 0
    errors = 0
    active = True

    def render() -> Group:
        idle = int(time.monotonic() - last)
        state = "Поток подключён" if active else "Поток завершён"
        header = Text(f"{state}  ·  {args.timezone}", style="bold cyan" if active else "yellow")
        header.append(f"\nВ видимом потоке: готово {finished} · предупреждений/ошибок {errors}")
        header.append(f"\nПоследнее событие {idle} с назад · Ctrl+C — остановить", style="dim")
        table = Table(box=box.SIMPLE, expand=True, padding=(0, 1))
        for name, width in (("Время", 8), ("Этап", None), ("ID", 7), ("Длит.", 7)):
            table.add_column(name, width=width, no_wrap=width is not None)
        for clock, style, label, mid, duration, details in list(rows)[
            -max(3, (console.height - 10) // 2) :
        ]:
            description = Text(label, style=style)
            if details:
                description.append("\n" + details, style="dim")
            table.add_row(Text(clock, style="dim"), description, Text(mid), Text(duration))
        return Group(Panel(header, title=Text(args.title), border_style="cyan"), table)

    try:
        with Live(get_renderable=render, console=console, refresh_per_second=2):
            for line in sys.stdin:
                row = parse_line(line, zone)
                rows.append(row)
                last = time.monotonic()
                finished += "voice_job_finished" in line
                errors += bool(re.search(r"\b(?:WARNING|ERROR|CRITICAL)\b", line))
            active = False
    except KeyboardInterrupt:
        console.print("Наблюдение остановлено.", style="dim")


if __name__ == "__main__":
    main()
