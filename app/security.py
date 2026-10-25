"""API-key auth + simple in-memory sliding-window rate limiter (single-process beta)."""
import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def api_key() -> str:
    return os.environ.get("AUDIO_ENGINE_API_KEY", "")


def check_key(request: Request):
    expected = api_key()
    if not expected:
        raise HTTPException(503, "server misconfigured: AUDIO_ENGINE_API_KEY not set")
    got = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(got.encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key")


LIMITS = {  # bucket -> (max calls, window seconds); override via env
    "separate": (int(os.environ.get("RATE_SEPARATE_PER_HOUR", "10")), 3600),
    "process": (int(os.environ.get("RATE_PROCESS_PER_HOUR", "60")), 3600),
}


class RateLimiter:
    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._hits.clear()

    def hit(self, bucket: str, ident: str):
        limit, window = LIMITS[bucket]
        now = time.monotonic()
        with self._lock:
            q = self._hits[(bucket, ident)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry = int(window - (now - q[0])) + 1
                raise HTTPException(429, f"rate limit: {limit} {bucket} calls/hour", headers={"Retry-After": str(retry)})
            q.append(now)


limiter = RateLimiter()


def client_ident(request: Request) -> str:
    # Single shared beta key, so limit per client IP (Render sets X-Forwarded-For) combined with key.
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    return f"{request.headers.get('x-api-key', '')[:8]}|{ip}"


def rate_limit(bucket: str):
    def dep(request: Request):
        limiter.hit(bucket, client_ident(request))
    return dep
