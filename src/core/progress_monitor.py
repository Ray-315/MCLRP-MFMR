from __future__ import annotations

import time


def log_stage(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[STAGE] {timestamp} | {message}", flush=True)


class ThrottledProgressLogger:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "items",
        min_interval_sec: float = 30.0,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.started_at = time.time()
        self.last_emit_at = 0.0
        self.last_value = -1

    def update(self, current: int, *, detail: str | None = None, force: bool = False) -> None:
        current = max(0, int(current))
        now = time.time()
        if not force:
            if current <= self.last_value:
                return
            if self.total > 0 and current < self.total and (now - self.last_emit_at) < self.min_interval_sec:
                return
        elapsed = now - self.started_at
        percent = 100.0 if self.total <= 0 else min(100.0, 100.0 * float(current) / float(self.total))
        parts = [
            f"[PROGRESS] {self.label}",
            f"{current}/{self.total} {self.unit}",
            f"({percent:.1f}%)",
            f"elapsed={elapsed / 60.0:.1f}m",
        ]
        if current > 0 and self.total > current:
            eta = elapsed * float(self.total - current) / float(current)
            parts.append(f"eta={eta / 60.0:.1f}m")
        if detail:
            parts.append(str(detail))
        print(" | ".join(parts), flush=True)
        self.last_emit_at = now
        self.last_value = current
