"""Model-neutral result and capability types.

Backends adapt their native output to these. Nothing outside a backend
module should ever see a model-specific object.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Segment:
    """One piece of recognised speech, times relative to the audio passed in.

    The attribute names match what ``ServeClientBase.update_segments`` reads
    (``text``, ``start``, ``end``, ``no_speech_prob``, ``words``), so the shared
    commit logic works on these directly.
    """
    text: str
    start: float = 0.0
    end: float = 0.0
    no_speech_prob: float = 0.0
    words: Optional[List[Word]] = None
    language: Optional[str] = None
    language_probability: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # model-specific labels (emotion, audio event ...)


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do. Sent to clients in the SERVER_READY handshake."""
    streaming: bool = False            # native streaming (accept/partial/is_endpoint) vs windowed re-transcribe
    translate: bool = False            # native X -> English translation (Whisper only)
    hotwords: bool = False
    initial_prompt: bool = False
    word_timestamps: bool = False
    language_detection: bool = False
    punctuation: bool = True           # False: output lacks punctuation/casing
    languages: Optional[Tuple[str, ...]] = None   # None = unrestricted or unknown
    sample_rate: int = 16000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["languages"] = list(self.languages) if self.languages is not None else None
        return d

    def supports_language(self, language: Optional[str]) -> bool:
        return (
            language is None
            or self.languages is None
            or language in ("auto",)
            or language in self.languages
        )
