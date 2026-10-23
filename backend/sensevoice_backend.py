"""SenseVoice-Small (FunAudioLLM/SenseVoiceSmall) as an offline backend.

Non-autoregressive, so one forward pass per window: cheap enough on CPU to re-run
on a growing window. It supports zh, en, yue, ja and ko with built-in language ID,
punctuation and inverse text normalisation. It has no hotwords, initial prompt,
word timestamps or translation, so the server routes those requests elsewhere.

Backend options (``--backend_opt key=value``):
    hub          "hf" (default, HuggingFace) or "ms" (ModelScope). Ignored for local paths.
    use_itn      true/false, inverse text normalisation (default true).
    cpu_threads  torch intra-op threads (default: leave torch's setting alone).
    vad_model    optional funasr VAD such as "fsmn-vad" for in-window splitting
                 (default none: the session already runs Silero before the model).
"""
import logging
import os
import re
from typing import List, Optional

import numpy as np

from whisper_live.engine.base import OfflineBackend
from whisper_live.engine.types import Capabilities, Segment

_TAG = re.compile(r"<\|([^|>]*)\|>")
_LANGS = ("zh", "en", "yue", "ja", "ko")
_EVENTS = {"speech", "bgm", "applause", "laughter", "cry", "sneeze", "breath", "cough"}
_EMOTIONS = {"happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised", "emo_unknown"}


def parse_sensevoice_text(raw: str) -> Segment:
    """``<|ja|><|NEUTRAL|><|Speech|><|withitn|>こんにちは。`` -> Segment(text='こんにちは。', language='ja')."""
    extra = {}
    language = None
    for tag in _TAG.findall(raw):
        low = tag.lower()
        if low in _LANGS:
            language = low
        elif low == "nospeech":
            extra["nospeech"] = True
        elif low in _EVENTS:
            extra["event"] = low
        elif low in _EMOTIONS:
            extra["emotion"] = low
    text = _TAG.sub("", raw).strip()
    no_speech = 1.0 if (extra.get("nospeech") or not text) else 0.0
    return Segment(text=text, language=language, no_speech_prob=no_speech, extra=extra)


class SenseVoiceBackend(OfflineBackend):
    CAPABILITIES = Capabilities(
        streaming=False, translate=False, hotwords=False, initial_prompt=False,
        word_timestamps=False, language_detection=True, punctuation=True,
        languages=_LANGS,
    )
    DEFAULT_LANGUAGE = "auto"

    def __init__(self):
        self.model = None
        self.use_itn = True

    def load(self, model: str = "FunAudioLLM/SenseVoiceSmall", device: str = "cpu", **opts) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:  # keep the hint next to the failure
            raise RuntimeError(
                "SenseVoice needs funasr: pip install -r requirements/sensevoice.txt"
            ) from e

        if opts.get("cpu_threads"):
            import torch
            torch.set_num_threads(int(opts["cpu_threads"]))
        self.use_itn = bool(opts.get("use_itn", True))

        kwargs = {"model": model, "device": device, "disable_update": True}
        if not os.path.isdir(model):
            kwargs["hub"] = opts.get("hub", "hf")
        if opts.get("vad_model"):
            kwargs["vad_model"] = opts["vad_model"]
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        logging.info(f"Loading SenseVoice: {model} on {device}")
        self.model = AutoModel(**kwargs)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   task: str = "transcribe", **opts) -> List[Segment]:
        if self.model is None:
            raise RuntimeError("SenseVoice model is not loaded")
        lang = language if language in _LANGS else "auto"
        audio = np.asarray(audio, dtype=np.float32)
        result = self.run_on_inference_thread(
            self.model.generate,
            input=audio, cache={}, language=lang, use_itn=self.use_itn,
            batch_size_s=60, disable_pbar=True,
        )
        duration = len(audio) / self.CAPABILITIES.sample_rate
        segments = []
        for item in result or []:
            seg = parse_sensevoice_text(item.get("text", ""))
            seg.start, seg.end = 0.0, duration   # SenseVoice gives no timings; the window is the unit
            segments.append(seg)
        return segments
