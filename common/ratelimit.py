from __future__ import annotations

import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from common import logger

_RATE_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}
_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", (text or "").strip().lower()).strip("-") or "limit"


def _header_token(text: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", (text or "").strip())
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _window_seconds_from_unit(unit: str) -> int:
    normalized = (unit or "").strip().lower()
    if normalized not in _RATE_UNITS:
        raise ValueError(f"Unsupported rate limit unit: {unit}")
    return _RATE_UNITS[normalized]


def _parse_size_bytes(value: str) -> int:
    normalized = (value or "").strip().lower()
    if not normalized:
        raise ValueError("Empty size string")
    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[a-z]+)?", normalized)
    if not match:
        raise ValueError(f"Invalid size string: {value}")
    number = float(match.group("number"))
    unit = (match.group("unit") or "b").lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unsupported size unit: {unit}")
    return max(0, int(number * _SIZE_UNITS[unit]))


@dataclass(frozen=True)
class FixedWindowRule:
    name: str
    scope: str
    limit: int
    window_seconds: int
    cost_name: str = "requests"
    status_code: int = 429

    @property
    def header_prefix(self) -> str:
        prefix = "Quota" if self.cost_name == "bytes" else "RateLimit"
        return f"X-{prefix}-{_header_token(self.name)}"

    @property
    def limit_spec(self) -> str:
        if self.cost_name == "bytes":
            return f"{self.limit}B/{self.window_seconds}s"
        return f"{self.limit}/{self.window_seconds}s"


@dataclass(frozen=True)
class LimitConsumption:
    used: int
    remaining: int
    reset_after_seconds: int


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    rule: FixedWindowRule
    consumption: LimitConsumption
    reason: str

    def headers(self) -> dict[str, str]:
        prefix = self.rule.header_prefix
        headers = {
            f"{prefix}-Limit": str(self.rule.limit),
            f"{prefix}-Remaining": str(max(0, self.consumption.remaining)),
            f"{prefix}-Reset": str(max(0, self.consumption.reset_after_seconds)),
            f"{prefix}-Policy": self.rule.limit_spec,
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, self.consumption.reset_after_seconds))
        return headers


@dataclass(frozen=True)
class RequestLimitEvaluation:
    allowed: bool
    decisions: list[LimitDecision] = field(default_factory=list)
    denied_decision: LimitDecision | None = None

    @property
    def headers(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for decision in self.decisions:
            merged.update(decision.headers())
        return merged

    def error_payload(self) -> dict[str, Any]:
        decision = self.denied_decision
        if decision is None:
            return {}
        return {
            "error": "rate limit exceeded" if decision.rule.cost_name == "requests" else "usage quota exceeded",
            "scope": decision.rule.scope,
            "rule": decision.rule.name,
            "cost_name": decision.rule.cost_name,
            "limit": decision.rule.limit,
            "window_seconds": decision.rule.window_seconds,
            "remaining": max(0, decision.consumption.remaining),
            "retry_after": max(1, decision.consumption.reset_after_seconds),
        }


@dataclass(frozen=True)
class AdmissionLease:
    pool: str


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    pool: str
    limit: int
    current: int
    retry_after_seconds: int = 1
    lease: AdmissionLease | None = None

    def headers(self) -> dict[str, str]:
        headers = {
            "X-Admission-Pool": self.pool,
            "X-Admission-Limit": str(max(0, self.limit)),
            "X-Admission-InFlight": str(max(0, self.current)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, self.retry_after_seconds))
        return headers

    def error_payload(self) -> dict[str, Any]:
        return {
            "error": "server busy",
            "pool": self.pool,
            "limit": max(0, self.limit),
            "inflight": max(0, self.current),
            "retry_after": max(1, self.retry_after_seconds),
        }


class RateLimitStore(ABC):
    backend_name = "unknown"

    @abstractmethod
    def consume(self, *, namespace: str, identity: str, limit: int, window_seconds: int, amount: int) -> LimitConsumption:
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryRateLimitStore(RateLimitStore):
    backend_name = "memory"

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict[str, tuple[int, float]] = {}
        self._ops = 0

    def _cleanup(self, now: float) -> None:
        expired_keys = [key for key, (_, expires_at) in self._values.items() if expires_at <= now]
        for key in expired_keys:
            self._values.pop(key, None)

    def consume(self, *, namespace: str, identity: str, limit: int, window_seconds: int, amount: int) -> LimitConsumption:
        now = time.time()
        bucket_start = int(now // window_seconds) * window_seconds
        expires_at = float(bucket_start + window_seconds)
        key = f"{namespace}:{bucket_start}:{identity}"
        with self._lock:
            self._ops += 1
            if self._ops % 1024 == 0:
                self._cleanup(now)
            current, _ = self._values.get(key, (0, expires_at))
            current += max(1, int(amount))
            self._values[key] = (current, expires_at)
        remaining = max(0, int(limit) - current)
        reset_after = max(1, int(expires_at - now))
        return LimitConsumption(used=current, remaining=remaining, reset_after_seconds=reset_after)

    def check_health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": self.backend_name, "entries": len(self._values)}


class RedisRateLimitStore(RateLimitStore):
    backend_name = "redis"

    def __init__(self, redis_url: str):
        if not redis_url:
            raise ValueError("DEEPDOC_RATE_LIMIT_REDIS_URL is required when backend=redis")
        try:
            import redis
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("redis package is required for Redis-backed rate limiting") from exc
        self.redis_url = redis_url
        self._client = redis.Redis.from_url(redis_url, decode_responses=False)

    def consume(self, *, namespace: str, identity: str, limit: int, window_seconds: int, amount: int) -> LimitConsumption:
        now = time.time()
        bucket_start = int(now // window_seconds) * window_seconds
        redis_key = f"deepdoc:ratelimit:{namespace}:{bucket_start}:{identity}"
        pipeline = self._client.pipeline()
        pipeline.incrby(redis_key, max(1, int(amount)))
        pipeline.ttl(redis_key)
        used, ttl_seconds = pipeline.execute()
        if int(ttl_seconds) < 0:
            self._client.expire(redis_key, int(window_seconds))
            ttl_seconds = int(window_seconds)
        remaining = max(0, int(limit) - int(used))
        return LimitConsumption(
            used=int(used),
            remaining=remaining,
            reset_after_seconds=max(1, int(ttl_seconds)),
        )

    def check_health(self) -> dict[str, Any]:
        self._client.ping()
        return {"status": "ok", "backend": self.backend_name, "url": self.redis_url}


class RequestRateLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        store: RateLimitStore,
        rules_by_scope: dict[str, list[FixedWindowRule]],
        fail_open: bool = True,
        admin_bypass: bool = True,
    ):
        self.enabled = bool(enabled)
        self.store = store
        self.rules_by_scope = {key: list(value or []) for key, value in (rules_by_scope or {}).items()}
        self.fail_open = bool(fail_open)
        self.admin_bypass = bool(admin_bypass)

    @property
    def backend_name(self) -> str:
        return getattr(self.store, "backend_name", "unknown")

    def evaluate(
        self,
        *,
        scope: str,
        identity: str,
        request_bytes: int = 0,
        is_admin: bool = False,
    ) -> RequestLimitEvaluation:
        if not self.enabled:
            return RequestLimitEvaluation(allowed=True)
        if self.admin_bypass and is_admin:
            return RequestLimitEvaluation(allowed=True)
        decisions: list[LimitDecision] = []
        for rule in self.rules_by_scope.get("general", []):
            decision = self._consume_rule(rule=rule, identity=identity, amount=1)
            decisions.append(decision)
            if not decision.allowed:
                return RequestLimitEvaluation(allowed=False, decisions=decisions, denied_decision=decision)
        for rule in self.rules_by_scope.get(scope, []):
            if rule.cost_name == "bytes":
                amount = int(request_bytes)
                if amount <= 0:
                    continue
            else:
                amount = 1
            decision = self._consume_rule(rule=rule, identity=identity, amount=amount)
            decisions.append(decision)
            if not decision.allowed:
                return RequestLimitEvaluation(allowed=False, decisions=decisions, denied_decision=decision)
        return RequestLimitEvaluation(allowed=True, decisions=decisions)

    def _consume_rule(self, *, rule: FixedWindowRule, identity: str, amount: int) -> LimitDecision:
        try:
            consumption = self.store.consume(
                namespace=_slug(rule.name),
                identity=identity,
                limit=rule.limit,
                window_seconds=rule.window_seconds,
                amount=amount,
            )
        except Exception:
            logger.exception("Rate limit backend failure rule=%s scope=%s", rule.name, rule.scope)
            if self.fail_open:
                return LimitDecision(
                    allowed=True,
                    rule=rule,
                    consumption=LimitConsumption(used=0, remaining=rule.limit, reset_after_seconds=0),
                    reason="backend-failed-open",
                )
            raise
        allowed = consumption.used <= rule.limit
        return LimitDecision(
            allowed=allowed,
            rule=rule,
            consumption=consumption,
            reason="ok" if allowed else "limit-exceeded",
        )

    def check_health(self) -> dict[str, Any]:
        state = self.store.check_health()
        state.update(
            {
                "enabled": self.enabled,
                "admin_bypass": self.admin_bypass,
                "fail_open": self.fail_open,
                "scopes": {
                    scope: [rule.limit_spec for rule in rules]
                    for scope, rules in self.rules_by_scope.items()
                    if rules
                },
            }
        )
        return state


class InflightAdmissionController:
    def __init__(self, max_by_pool: dict[str, int] | None = None):
        self.max_by_pool = {key: max(0, int(value)) for key, value in (max_by_pool or {}).items() if int(value) > 0}
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {key: 0 for key in self.max_by_pool}

    @property
    def enabled(self) -> bool:
        return bool(self.max_by_pool)

    def acquire(self, pool: str) -> AdmissionDecision:
        limit = max(0, int(self.max_by_pool.get(pool, 0)))
        if limit <= 0:
            return AdmissionDecision(allowed=True, pool=pool, limit=0, current=0, lease=None)
        with self._lock:
            current = int(self._counts.get(pool, 0))
            if current >= limit:
                return AdmissionDecision(allowed=False, pool=pool, limit=limit, current=current)
            current += 1
            self._counts[pool] = current
        return AdmissionDecision(
            allowed=True,
            pool=pool,
            limit=limit,
            current=current,
            lease=AdmissionLease(pool=pool),
        )

    def release(self, lease: AdmissionLease | None) -> None:
        if lease is None:
            return
        pool = lease.pool
        if not pool:
            return
        with self._lock:
            current = int(self._counts.get(pool, 0))
            if current <= 1:
                self._counts[pool] = 0
            else:
                self._counts[pool] = current - 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "limits": dict(self.max_by_pool),
                "inflight": dict(self._counts),
            }


def parse_rate_rule(name: str, scope: str, spec: str | None) -> FixedWindowRule | None:
    text = str(spec or "").strip()
    if not text or text.lower() in {"0", "none", "off", "disabled"}:
        return None
    match = re.fullmatch(r"(?P<count>\d+)\s*/\s*(?P<unit>[A-Za-z]+)", text)
    if not match:
        raise ValueError(f"Invalid rate limit spec for {name}: {spec}")
    return FixedWindowRule(
        name=name,
        scope=scope,
        limit=max(1, int(match.group("count"))),
        window_seconds=_window_seconds_from_unit(match.group("unit")),
        cost_name="requests",
    )


def parse_bytes_rule(name: str, scope: str, spec: str | None) -> FixedWindowRule | None:
    text = str(spec or "").strip()
    if not text or text.lower() in {"0", "none", "off", "disabled"}:
        return None
    match = re.fullmatch(r"(?P<size>\d+(?:\.\d+)?\s*[A-Za-z]*)\s*/\s*(?P<unit>[A-Za-z]+)", text)
    if not match:
        raise ValueError(f"Invalid byte quota spec for {name}: {spec}")
    return FixedWindowRule(
        name=name,
        scope=scope,
        limit=max(1, _parse_size_bytes(match.group("size"))),
        window_seconds=_window_seconds_from_unit(match.group("unit")),
        cost_name="bytes",
    )


def create_request_rate_limiter() -> RequestRateLimiter:
    enabled = _parse_bool(os.environ.get("DEEPDOC_RATE_LIMIT_ENABLED"), default=False)
    backend = (os.environ.get("DEEPDOC_RATE_LIMIT_BACKEND") or "memory").strip().lower()
    if backend == "redis":
        store: RateLimitStore = RedisRateLimitStore(os.environ.get("DEEPDOC_RATE_LIMIT_REDIS_URL", ""))
    else:
        store = InMemoryRateLimitStore()
    rules: dict[str, list[FixedWindowRule]] = {"general": []}
    configured_rules = [
        ("general", parse_rate_rule("general", "general", os.environ.get("DEEPDOC_RATE_LIMIT_GENERAL"))),
        ("parse", parse_rate_rule("parse", "parse", os.environ.get("DEEPDOC_RATE_LIMIT_PARSE"))),
        ("parse", parse_bytes_rule("parse-bytes", "parse", os.environ.get("DEEPDOC_RATE_LIMIT_PARSE_BYTES"))),
        ("artifact", parse_rate_rule("artifact", "artifact", os.environ.get("DEEPDOC_RATE_LIMIT_ARTIFACT"))),
        ("ingest", parse_rate_rule("ingest", "ingest", os.environ.get("DEEPDOC_RATE_LIMIT_INGEST"))),
        ("admin", parse_rate_rule("admin", "admin", os.environ.get("DEEPDOC_RATE_LIMIT_ADMIN"))),
    ]
    for scope, rule in configured_rules:
        if rule is None:
            continue
        rules.setdefault(scope, []).append(rule)
    return RequestRateLimiter(
        enabled=enabled,
        store=store,
        rules_by_scope=rules,
        fail_open=_parse_bool(os.environ.get("DEEPDOC_RATE_LIMIT_FAIL_OPEN"), default=True),
        admin_bypass=_parse_bool(os.environ.get("DEEPDOC_RATE_LIMIT_ADMIN_BYPASS"), default=True),
    )


def create_inflight_admission_controller() -> InflightAdmissionController:
    limits = {
        "parse": int(os.environ.get("DEEPDOC_MAX_INFLIGHT_PARSE", "0") or 0),
        "artifact": int(os.environ.get("DEEPDOC_MAX_INFLIGHT_ARTIFACT", "0") or 0),
        "ingest": int(os.environ.get("DEEPDOC_MAX_INFLIGHT_INGEST", "0") or 0),
    }
    return InflightAdmissionController(max_by_pool=limits)
