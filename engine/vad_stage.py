"""Session-level voice activity detection.

Runs before any backend, so it is model independent. Offline models use it to
skip silence and to commit on a pause; streaming models use it as a fallback
endpoint signal. All times are seconds on the session's absolute audio clock.
"""
import threading
from typing import Optional

import numpy as np

_SHARED_DETECTOR = None
_SHARED_LOCK = threading.Lock()


def get_shared_detector(threshold: float = 0.5):
    """One Silero detector per process (``audio_forward`` resets its state every call)."""
    global _SHARED_DETECTOR
    with _SHARED_LOCK:
        if _SHARED_DETECTOR is None:
            from whisper_live.vad import VoiceActivityDetector
            _SHARED_DETECTOR = VoiceActivityDetector(threshold=threshold)
        return _SHARED_DETECTOR


class SessionVAD:
    MIN_SAMPLES = 512

    def __init__(self, detector=None, sample_rate: int = 16000, threshold: float = 0.5):
        self._detector = detector
        self._threshold = threshold
        self.sample_rate = sample_rate
        self.total_s = 0.0           # audio received so far
        self.last_speech_s = -1.0    # end time of the most recent speech frame
        self._tail = np.zeros(0, dtype=np.float32)   # samples not yet long enough to classify
        self._lock = threading.Lock()

    @property
    def detector(self):
        if self._detector is None:
            self._detector = get_shared_detector(self._threshold)
        return self._detector

    def update(self, frame: np.ndarray) -> bool:
        """Classify one received frame and advance the clock."""
        real_len = len(frame)
        # Silero needs a whole 512-sample window. Padding a short frame with zeros dilutes the speech
        # (a 10 ms frame would never register), so short frames wait here until enough have arrived.
        if self._tail.size:
            frame = np.concatenate([self._tail, frame])
        if len(frame) < self.MIN_SAMPLES:
            self._tail = frame
            speech = False
        else:
            self._tail = frame[:0]
            speech = bool(self.detector(frame))
        with self._lock:
            self.total_s += real_len / self.sample_rate
            if speech:
                self.last_speech_s = self.total_s
        return speech

    def speech_since(self, t: float) -> bool:
        """Was there speech after absolute time ``t``?"""
        with self._lock:
            return self.last_speech_s > t

    def silence_s(self) -> float:
        with self._lock:
            return self.total_s - self.last_speech_s if self.last_speech_s >= 0 else 0.0

    def endpoint_at(self, committed_s: float, upto_s: float, min_silence_s: float) -> bool:
        """Like ``endpoint`` but judged on audio up to ``upto_s`` only.

        A window is a snapshot; speech that arrived after it was taken is not in it, so the
        utterance only counts as finished when the snapshot itself ends in enough silence.
        """
        with self._lock:
            ls = self.last_speech_s
            return committed_s < ls <= upto_s and upto_s - ls >= min_silence_s

    def endpoint(self, committed_s: float, min_silence_s: float) -> bool:
        """Speech happened after the last commit and has been followed by enough silence."""
        with self._lock:
            return (self.last_speech_s > committed_s
                    and self.total_s - self.last_speech_s >= min_silence_s)
