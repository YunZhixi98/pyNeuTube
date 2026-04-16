from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import pyneutube.tracing as tracing


class _CollectingQueue:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, int | None, int | None]] = []

    def put(self, item: tuple[str, str, int | None, int | None]) -> None:
        self.items.append(item)


class _FakeTqdm:
    desc_refresh_true_calls = 0
    refresh_calls = 0

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc", "")
        self.disable = kwargs.get("disable", False)
        self.n = 0

    @classmethod
    def reset(cls) -> None:
        cls.desc_refresh_true_calls = 0
        cls.refresh_calls = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        self.desc = "" if desc is None else desc
        if refresh:
            type(self).desc_refresh_true_calls += 1

    def set_postfix_str(self, value: str) -> None:
        self.desc = value

    def refresh(self) -> None:
        type(self).refresh_calls += 1

    def write(self, message: str) -> None:
        del message

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


def _fake_trace_file_worker(
    payload: tuple[str, str, str | None, int, float | None, int, bool, str | None, object | None],
) -> dict[str, object]:
    input_path, output_swc, _visualization_dir, _n_jobs, _timeout, _verbose, _overwrite, _config, progress_queue = (
        payload
    )
    if progress_queue is not None:
        progress_queue.put((input_path, "load_image", None, None))
        progress_queue.put((input_path, "generate_tracing_seeds", 0, 240))
        for current in range(1, 241):
            progress_queue.put((input_path, "generate_tracing_seeds", current, 240))
        progress_queue.put((input_path, "reconstruct", None, None))
    return {
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "status": "completed",
        "input_path": input_path,
        "output_swc": output_swc,
        "elapsed_seconds": 0.01,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "timeout_seconds": None,
    }


def test_queued_batch_progress_reporter_throttles_repeated_updates() -> None:
    queue = _CollectingQueue()
    reporter = tracing._QueuedBatchProgressReporter(
        "sample.v3dpbd",
        queue,
        refresh_every=50,
        min_interval=999.0,
        timer=lambda: 0.0,
    )

    reporter.emit("generate_tracing_seeds", 0, 240)
    for current in range(1, 151):
        reporter.emit("generate_tracing_seeds", current, 240)
    reporter.emit("reconstruct", None, None)
    reporter.emit("generate_tracing_seeds", 240, 240)

    assert queue.items == [
        ("sample.v3dpbd", "generate_tracing_seeds", 0, 240),
        ("sample.v3dpbd", "generate_tracing_seeds", 50, 240),
        ("sample.v3dpbd", "generate_tracing_seeds", 100, 240),
        ("sample.v3dpbd", "generate_tracing_seeds", 150, 240),
        ("sample.v3dpbd", "reconstruct", None, None),
        ("sample.v3dpbd", "generate_tracing_seeds", 240, 240),
    ]


def test_trace_files_batch_progress_does_not_force_refresh_per_message(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_paths = []
    for index in range(3):
        input_path = tmp_path / f"sample_{index}.tif"
        input_path.write_bytes(b"fake")
        input_paths.append(input_path)

    _FakeTqdm.reset()
    monkeypatch.setattr(tracing, "tqdm", _FakeTqdm)
    monkeypatch.setattr(tracing, "_trace_file_worker", _fake_trace_file_worker)
    monkeypatch.setattr(tracing, "_batch_pool_context", lambda show_progress: get_context("fork"))

    tracing.trace_files(
        input_paths,
        tmp_path / "out",
        batch_n_jobs=2,
        trace_n_jobs=1,
        verbose=1,
        overwrite=True,
    )

    assert _FakeTqdm.desc_refresh_true_calls == 0
    assert _FakeTqdm.refresh_calls < 40


def test_batch_pool_context_uses_spawn_when_progress_is_enabled() -> None:
    if tracing.os.name != "posix":
        return
    assert tracing._batch_pool_context(show_progress=True).get_start_method() == "spawn"
    assert tracing._batch_pool_context(show_progress=False).get_start_method() == "fork"
