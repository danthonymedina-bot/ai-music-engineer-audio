"""Tiny in-process job store (JSON on disk so status survives restarts of the reader, not the worker)."""
import json
import threading
import time
import uuid
from pathlib import Path
from .config import STORAGE_DIR

_lock = threading.Lock()


def job_dir(job_id: str) -> Path:
    return STORAGE_DIR / "jobs" / job_id


def create(kind: str, params: dict) -> str:
    jid = uuid.uuid4().hex
    d = job_dir(jid)
    d.mkdir(parents=True)
    _write(jid, dict(id=jid, kind=kind, status="queued", params=params, outputs={}, error=None,
                     created_at=time.time(), updated_at=time.time()))
    return jid


def _write(jid, data):
    p = job_dir(jid) / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(p)


def get(jid: str):
    p = job_dir(jid) / "job.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def update(jid: str, **kw):
    with _lock:
        d = get(jid)
        d.update(kw, updated_at=time.time())
        _write(jid, d)
        return d
