"""Structured JSON logging via structlog, with credential redaction.

All output, stdlib logging from third-party libraries included, renders as one JSON stream on
stdout, so every trade decision can be reconstructed from the logs. Call configure_logging()
once at process start.
"""

import logging
import re
import sys

import structlog

# FRED sends its key as a URL query param. The httpx/httpcore level pin in configure_logging
# silences the known emitter; this catches the rest (a library logging its request line,
# exception text we failed to sanitize). Any param NAME that looks like a credential matches,
# with an optional `x_cg_pro_`-style prefix that must end in _ or - so "monkey=" is not a key.
_URL_SECRET_RE = re.compile(
    r"(?<![\w-])((?:[\w-]*[_-])?(?:api[_-]?key|apikey|key|token|secret|auth|password"
    r"|sig|signature))"
    r"=[^&\s\"'<>]+",
    re.IGNORECASE,
)


def _scrub(value: object) -> object:
    """Rewrite credential query params inside a logged value to `<name>=REDACTED`.

    Containers are rebuilt; other objects are checked through their repr (what JSONRenderer
    emits) and replaced by it only when it held a secret.
    """
    if isinstance(value, str):
        return _URL_SECRET_RE.sub(r"\1=REDACTED", value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, dict):
        # Keys too. A non-string key becomes str: json.dumps cannot serialize a tuple key,
        # and the handler would then drop the whole record.
        return {_scrub(k if isinstance(k, str) else str(k)): _scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    rendered = repr(value)  # JSONRenderer falls back to repr() for anything else
    scrubbed = _URL_SECRET_RE.sub(r"\1=REDACTED", rendered)
    return scrubbed if scrubbed != rendered else value


def redact_url_secrets(logger: object, method_name: str, event_dict: dict) -> dict:
    """structlog processor: scrub credential query params out of every field, in place.

    A value that cannot be scrubbed is replaced by a marker, never passed through.
    """
    for key, value in event_dict.items():
        try:
            event_dict[key] = _scrub(value)
        except Exception as e:
            # Raising here would escape log.warning() into the caller. Dropping the field
            # fails closed: an unscrubbable value is exactly the one that might hold a secret.
            event_dict[key] = f"<redaction failed: {type(e).__name__}>"
    return event_dict


# Error-level events of a cycle, drained by the notifier node into the Discord alert, so
# log.error / log.exception PAGES A HUMAN; degradations the cycle absorbs by design are
# warnings. Only our own events are kept (third-party text can carry URLs and bodies), and
# overflow past the cap is counted and reported.
ALERT_LEVELS = frozenset({"error", "critical"})
ALERT_BUFFER_MAX = 50
_ALERTS: list[dict] = []
_DROPPED = 0


def collect_alerts(logger: object, method_name: str, event_dict: dict) -> dict:
    """structlog processor: keep scalar fields of our error-level events for alerting."""
    global _DROPPED
    if event_dict.get("level") not in ALERT_LEVELS or event_dict.get("_from_structlog") is False:
        return event_dict
    if len(_ALERTS) >= ALERT_BUFFER_MAX:
        _DROPPED += 1
        return event_dict
    _ALERTS.append({k: v for k, v in event_dict.items() if isinstance(v, str | int | float | bool)})
    return event_dict


def drain_alerts() -> list[dict]:
    """Return and clear the collected events, plus an `alerts_dropped` count on overflow."""
    global _DROPPED
    items = list(_ALERTS)
    if _DROPPED:
        items.append({"event": "alerts_dropped", "level": "error", "count": _DROPPED})
    _ALERTS.clear()
    _DROPPED = 0
    return items


_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    # Before collect_alerts, so the redacted value is what reaches Discord too.
    redact_url_secrets,
    collect_alerts,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
]


def configure_logging(level: int = logging.INFO) -> None:
    """Route structlog and the stdlib root logger into one JSON stream on stdout."""
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            redact_url_secrets,  # again: the traceback only becomes a string just now
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # httpx logs every request URL at INFO (httpcore at DEBUG), and FRED's key is in its URL,
    # so both stay at WARNING whatever the root level.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)
