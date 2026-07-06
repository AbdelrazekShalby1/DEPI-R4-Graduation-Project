import asyncio
import logging
import threading
import time
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
logger = logging.getLogger(__name__)

COOLDOWN_SECONDS     = 60
MAX_COOLDOWN_SECONDS = 300


class KeyManager:
    def __init__(self, keys: list[str]):
        self._keys:      list[str]        = list(keys)
        self._index:     int              = 0
        self._cooldowns: dict[str, float] = {}
        self._lock       = threading.Lock()

    def reload(self, keys: list[str]) -> None:
        with self._lock:
            self._keys  = list(keys)
            self._index = 0
        logger.info("KeyManager: loaded %d key(s)", len(keys))

    def get(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("No API keys configured")
            self._prune_cooldowns()
            now   = time.time()
            total = len(self._keys)
            for _ in range(total):
                key         = self._keys[self._index % total]
                self._index = (self._index + 1) % total
                if now >= self._cooldowns.get(key, 0):
                    return key
            key = min(self._keys, key=lambda k: self._cooldowns.get(k, 0))
            logger.warning("All keys cooling — using soonest: ...%s", key[-6:])
            return key

    def get_client(self) -> tuple[genai.Client, str]:
        key = self.get()
        return genai.Client(api_key=key), key

    def mark_rate_limited(self, key: str,
                          retry_after: int = COOLDOWN_SECONDS) -> None:
        with self._lock:
            if key not in self._keys:
                logger.debug(
                    "mark_rate_limited called for unknown/removed key ...%s — ignored",
                    key[-6:],
                )
                return
            now          = time.time()
            remaining    = max(0.0, self._cooldowns.get(key, 0) - now)
            backoff      = max(retry_after, remaining * 2) if remaining > 0 else retry_after
            new_cooldown = min(backoff, MAX_COOLDOWN_SECONDS)
            until        = now + new_cooldown
            self._cooldowns[key] = until
            logger.warning(
                "Key ...%s rate-limited — cooling for %ds (until %s)",
                key[-6:], int(new_cooldown),
                time.strftime("%H:%M:%S", time.localtime(until)),
            )

    def mark_invalid(self, key: str) -> None:
        with self._lock:
            before     = len(self._keys)
            self._keys = [k for k in self._keys if k != key]
            removed    = before - len(self._keys)
            if removed:
                logger.error("Key ...%s is invalid — removed from rotation", key[-6:])
            if not self._keys:
                self._index = 0
                logger.error("KeyManager: no valid keys remaining!")

    def get_current_key(self) -> str:
        return self.get()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    @property
    def keys(self) -> list[str]:
        with self._lock:
            return list(self._keys)

    def _prune_cooldowns(self) -> None:
        now = time.time()
        self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > now}


def _load_keys_from_env() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_KEY_{i}")
        if not key:
            break
        keys.append(key.strip())
        i += 1
    if not keys:
        raise ValueError("مفيش Keys في الـ .env! حطي GEMINI_KEY_1 على الأقل")

    if len(set(keys)) < len(keys):
        logger.warning(
            "⚠️ في مفاتيح مكررة في .env (نفس المفتاح أكتر من مرة). "
            "الـ rotation مش هيفيد فعلياً — استخدم مفاتيح مختلفة."
        )
    return keys


key_manager = KeyManager(_load_keys_from_env())


def is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    s = str(exc).lower()
    return any(x in s for x in ["429", "quota", "rate limit", "resource exhausted"])


def is_invalid_key_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (401, 403):
        return True
    s = str(exc).lower()
    return "401" in s or "403" in s or "api key not valid" in s or "invalid" in s


def is_transient_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (500, 502, 503, 504):
        return True
    s = str(exc).lower()
    return any(x in s for x in [
        "503", "502", "500", "504",
        "unavailable", "high demand", "internal error",
        "overloaded", "try again later",
    ])


TRANSIENT_MAX_RETRIES  = 3
TRANSIENT_BACKOFF_BASE = 1.0


def with_key_retry(call, *, max_retries: int | None = None):
    max_retries = max_retries or (key_manager.count + 1)
    last_error  = None
    key_attempt = 0
    transient_attempt = 0

    while True:
        client, key = key_manager.get_client()
        try:
            result = call(client)
            return result, key
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                logger.warning("Rate limit on key ...%s (attempt %d)", key[-6:], key_attempt + 1)
                key_manager.mark_rate_limited(key)
                key_attempt += 1
                if key_attempt >= max_retries:
                    break
                continue
            elif is_invalid_key_error(e):
                logger.error("Invalid key ...%s", key[-6:])
                key_manager.mark_invalid(key)
                key_attempt += 1
                if key_attempt >= max_retries:
                    break
                continue
            elif is_transient_error(e):
                transient_attempt += 1
                if transient_attempt > TRANSIENT_MAX_RETRIES:
                    break
                backoff = min(TRANSIENT_BACKOFF_BASE * (2 ** (transient_attempt - 1)), 8)
                logger.warning(
                    "Transient server error (attempt %d/%d) — retrying in %.1fs: %s",
                    transient_attempt, TRANSIENT_MAX_RETRIES, backoff, e,
                )
                time.sleep(backoff)
                continue
            raise

    raise last_error or Exception("All API keys failed")


async def with_key_retry_async(call, *, max_retries: int | None = None):
    max_retries = max_retries or (key_manager.count + 1)
    last_error  = None
    key_attempt = 0
    transient_attempt = 0

    while True:
        client, key = key_manager.get_client()
        try:
            result = await call(client)
            return result, key
        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                logger.warning("Rate limit on key ...%s (attempt %d)", key[-6:], key_attempt + 1)
                key_manager.mark_rate_limited(key)
                key_attempt += 1
                if key_attempt >= max_retries:
                    break
                continue
            elif is_invalid_key_error(e):
                logger.error("Invalid key ...%s", key[-6:])
                key_manager.mark_invalid(key)
                key_attempt += 1
                if key_attempt >= max_retries:
                    break
                continue
            elif is_transient_error(e):
                transient_attempt += 1
                if transient_attempt > TRANSIENT_MAX_RETRIES:
                    break
                backoff = min(TRANSIENT_BACKOFF_BASE * (2 ** (transient_attempt - 1)), 8)
                logger.warning(
                    "Transient server error (attempt %d/%d) — retrying in %.1fs: %s",
                    transient_attempt, TRANSIENT_MAX_RETRIES, backoff, e,
                )
                await asyncio.sleep(backoff)
                continue
            raise

    raise last_error or Exception("All API keys failed")
