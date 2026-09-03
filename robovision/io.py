"""File helpers for training and evaluation outputs."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import signal
import threading


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def stop_after_update():
    requested = [False]
    handlers = {}

    def request_stop(signum, frame):
        requested[0] = True

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, request_stop)
    try:
        yield requested
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
