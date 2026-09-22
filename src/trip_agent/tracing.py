"""Thin Langfuse wrapper that degrades to a no-op when keys aren't configured.

Tracing being optional matters: the eval scores the local trajectory, so tests
and offline runs work without a Langfuse account. With keys set, each run gets
one trace (agent span -> generation per Claude turn -> tool span per call).
"""

from __future__ import annotations

import os
from contextlib import contextmanager


class _NoopObservation:
    def update(self, **_kwargs):
        return self


class Tracer:
    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
        self.enabled = enabled
        self.client = None
        if enabled:
            # Keep eval traffic separate from any other use of the same Langfuse project.
            os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")
            from langfuse import get_client

            self.client = get_client()

    @contextmanager
    def trace(self, name: str, *, session_id: str | None = None, tags: list[str] | None = None,
              metadata: dict | None = None, input=None):
        """Root observation for one agent run. Yields (observation, trace_id)."""
        if not self.enabled:
            yield _NoopObservation(), None
            return
        from langfuse import propagate_attributes

        with propagate_attributes(trace_name=name, session_id=session_id, tags=tags,
                                  metadata={k: str(v) for k, v in (metadata or {}).items()}):
            with self.client.start_as_current_observation(as_type="agent", name=name, input=input) as root:
                yield root, self.client.get_current_trace_id()

    @contextmanager
    def observation(self, **kwargs):
        if not self.enabled:
            yield _NoopObservation()
            return
        with self.client.start_as_current_observation(**kwargs) as obs:
            yield obs

    def current_context(self):
        """Capture the active trace context so worker threads nest under the same parent."""
        if not self.enabled:
            return None
        from opentelemetry import context as otel_context

        return otel_context.get_current()

    @contextmanager
    def attached(self, ctx):
        if ctx is None:
            yield
            return
        from opentelemetry import context as otel_context

        token = otel_context.attach(ctx)
        try:
            yield
        finally:
            otel_context.detach(token)

    def score(self, trace_id: str | None, name: str, value, comment: str | None = None,
              data_type: str | None = None):
        if not (self.enabled and trace_id):
            return
        self.client.create_score(trace_id=trace_id, name=name, value=value, comment=comment,
                                 data_type=data_type)

    def trace_url(self, trace_id: str | None) -> str | None:
        if not (self.enabled and trace_id):
            return None
        return self.client.get_trace_url(trace_id=trace_id)

    def flush(self):
        if self.enabled:
            self.client.flush()
