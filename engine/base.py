"""Backend interfaces.

Offline backends transcribe a whole audio window per call and are driven by
``WindowedSession``. Streaming backends keep internal state and are driven by
``StreamingSession``.
"""
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from whisper_live.engine.types import Capabilities, Segment


_EXECUTOR_LOCK = threading.Lock()


class OfflineBackend(ABC):
    """Transcribes a float32 mono 16 kHz window. Instances are shared by all sessions."""

    CAPABILITIES: Capabilities = Capabilities()
    #: language used when the client does not name one (None = backend cannot auto-detect)
    DEFAULT_LANGUAGE: Optional[str] = None
    #: set False when transcribe() may be called from several threads at once
    NEEDS_LOCK: bool = True

    @property
    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def run_on_inference_thread(self, fn, *args, **kwargs):
        """Call ``fn`` on one long-lived worker thread and wait for the result.

        Each session decodes from its own thread, and torch keeps per-thread state it never gives
        back: calling a torch model from a fresh thread every session grew the server by ~19 MB per
        session. Funnelling every call through one thread keeps memory flat (and the calls were
        already serialised by the shared model lock). Backends built on torch should use this.
        """
        executor = self.__dict__.get("_inference_executor")
        if executor is None:
            with _EXECUTOR_LOCK:
                executor = self.__dict__.get("_inference_executor")
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=type(self).__name__)
                    self._inference_executor = executor
        return executor.submit(fn, *args, **kwargs).result()

    @abstractmethod
    def load(self, model: str, device: str = "cpu", **opts) -> None:
        """Load weights. ``model`` is a local path or hub id; ``opts`` come from --backend_opt."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   task: str = "transcribe", **opts) -> List[Segment]:
        """Return segments with times relative to the start of ``audio``."""

    def warmup(self) -> None:
        """Run one tiny inference so the first real request is not slow."""
        self.transcribe(np.zeros(16000, dtype=np.float32))


class StreamingBackend(ABC):
    """Stateful recogniser fed chunk by chunk. One instance per session.

    Heavy weights should be cached at class level so ``load`` is cheap for the
    second and later sessions.
    """

    CAPABILITIES: Capabilities = Capabilities(streaming=True)
    DEFAULT_LANGUAGE: Optional[str] = None

    @property
    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    @abstractmethod
    def load(self, model: str, device: str = "cpu", **opts) -> None:
        """Load (or fetch the cached) recogniser and open a fresh stream."""

    @abstractmethod
    def accept(self, chunk: np.ndarray) -> None:
        """Feed float32 mono 16 kHz samples."""

    @abstractmethod
    def partial(self) -> Optional[Segment]:
        """Current unfinished hypothesis, or None when there is nothing yet."""

    @abstractmethod
    def is_endpoint(self) -> bool:
        """True when the model itself decides the utterance has ended."""

    @abstractmethod
    def reset(self) -> None:
        """Start a new utterance (state after an endpoint)."""

    def final(self) -> Optional[Segment]:
        """Final text for the utterance that just ended. Default: the last partial."""
        return self.partial()
