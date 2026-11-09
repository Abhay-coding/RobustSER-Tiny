"""
Distributed tracing support for RobustSER-Tiny.

Provides trace context propagation for correlating operations across
training, evaluation, and inference.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Any
from contextlib import contextmanager


# Context variable for trace context
_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar("trace_context", default=None)


@dataclass
class TraceContext:
    """
    Trace context for correlating operations.

    Fields follow W3C TraceContext standard:
    - trace_id: 16-byte identifier for the entire trace (32 hex chars)
    - span_id: 8-byte identifier for the current span (16 hex chars)
    - trace_flags: flags (0x01 = sampled)
    """
    trace_id: str
    span_id: str
    trace_flags: int = 1  # sampled by default
    parent_span_id: Optional[str] = None
    baggage: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(cls, parent: Optional["TraceContext"] = None) -> "TraceContext":
        """Create a new trace context, optionally as child of parent."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = parent.span_id if parent else None
        baggage = parent.baggage.copy() if parent else {}
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["TraceContext"]:
        """Create trace context from W3C traceparent header."""
        traceparent = headers.get("traceparent") or headers.get("Traceparent")
        if not traceparent:
            return None

        # Format: version-trace-id-span-id-flags
        parts = traceparent.split("-")
        if len(parts) != 4:
            return None

        version, trace_id, span_id, flags = parts
        if version != "00":
            return None

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=int(flags, 16),
        )

    def to_headers(self) -> dict[str, str]:
        """Convert to W3C trace headers."""
        headers = {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}",
        }
        if self.baggage:
            baggage_header = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["baggage"] = baggage_header
        return headers

    def child(self) -> "TraceContext":
        """Create a child span."""
        return TraceContext.new(parent=self)

    def with_baggage(self, key: str, value: str) -> "TraceContext":
        """Add baggage (propagated key-value pairs)."""
        new_baggage = self.baggage.copy()
        new_baggage[key] = value
        return TraceContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            trace_flags=self.trace_flags,
            parent_span_id=self.parent_span_id,
            baggage=new_baggage,
        )


def get_trace_context() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _trace_context.get()


def set_trace_context(context: Optional[TraceContext]) -> None:
    """Set the current trace context."""
    _trace_context.set(context)


@contextmanager
def trace_span(
    name: str,
    parent: Optional[TraceContext] = None,
    baggage: Optional[dict[str, str]] = None,
) -> "TraceSpan":
    """
    Context manager for a trace span.

    Usage:
        with trace_span("train_epoch", baggage={"epoch": "1"}) as span:
            span.set_attribute("loss", 0.5)
            # do work
    """
    current = get_trace_context()
    parent_ctx = parent or current
    span_ctx = TraceContext.new(parent=parent_ctx)

    if baggage:
        for k, v in baggage.items():
            span_ctx = span_ctx.with_baggage(k, v)

    token = _trace_context.set(span_ctx)
    span = TraceSpan(span_ctx, name)

    try:
        yield span
    except Exception as e:
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(e))
        raise
    finally:
        _trace_context.reset(token)


@dataclass
class TraceSpan:
    """A trace span with attributes and events."""
    context: TraceContext
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=lambda: __import__("time").time())
    end_time: Optional[float] = None

    def set_attribute(self, key: str, value: Any) -> "TraceSpan":
        """Set a span attribute."""
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> "TraceSpan":
        """Add an event to the span."""
        import time
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
        return self

    def end(self) -> None:
        """Mark span as ended."""
        import time
        self.end_time = time.time()

    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "attributes": self.attributes,
            "events": self.events,
        }


# Convenience function for creating trace context from correlation IDs
def trace_context_from_correlation(
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    epoch: Optional[int] = None,
    seed: Optional[int] = None,
    split: Optional[str] = None,
) -> TraceContext:
    """Create a trace context with correlation IDs as baggage."""
    ctx = TraceContext.new()
    baggage = {}
    if experiment_id:
        baggage["experiment_id"] = experiment_id
    if run_id:
        baggage["run_id"] = run_id
    if epoch is not None:
        baggage["epoch"] = str(epoch)
    if seed is not None:
        baggage["seed"] = str(seed)
    if split:
        baggage["split"] = split

    if baggage:
        ctx = TraceContext(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            trace_flags=ctx.trace_flags,
            parent_span_id=ctx.parent_span_id,
            baggage=baggage,
        )
    return ctx