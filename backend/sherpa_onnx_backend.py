"""sherpa-onnx online (streaming transducer) recogniser as a native streaming backend.

``model`` is a directory holding ``tokens.txt`` plus encoder/decoder/joiner ONNX files
(int8 preferred when present). Runs on CPU. Heavy weights are cached per
(directory, options) so every session shares one ``OnlineRecognizer`` and only owns a stream.

Backend options: ``encoder``, ``decoder``, ``joiner``, ``tokens`` (file names to override
auto-discovery), ``num_threads`` (default 2), ``decoding_method`` (default greedy_search),
``rule1``/``rule2``/``rule3`` endpoint rules in seconds (defaults 2.4 / 1.2 / 20.0),
``provider`` (default cpu).

Output of many of these models has no punctuation or casing; see Capabilities.punctuation.
"""
import glob
import os
import threading
from typing import Dict, Optional

import numpy as np

from whisper_live.engine.base import StreamingBackend
from whisper_live.engine.types import Capabilities, Segment

_RECOGNIZERS: Dict[tuple, object] = {}
_LOCK = threading.Lock()


def _pick(model_dir: str, stem: str, override: Optional[str]) -> str:
    if override:
        return override if os.path.isabs(override) else os.path.join(model_dir, override)
    int8 = sorted(glob.glob(os.path.join(model_dir, f"{stem}*int8*.onnx")))
    plain = sorted(f for f in glob.glob(os.path.join(model_dir, f"{stem}*.onnx")) if "int8" not in f)
    found = int8 or plain
    if not found:
        raise FileNotFoundError(f"No {stem}*.onnx in {model_dir}")
    return found[0]


class SherpaOnnxOnlineBackend(StreamingBackend):
    CAPABILITIES = Capabilities(
        streaming=True, translate=False, hotwords=False, initial_prompt=False,
        word_timestamps=False, language_detection=False, punctuation=False,
    )

    def __init__(self):
        self._recognizer = None
        self._stream = None
        self._sample_rate = 16000

    def load(self, model: str, device: str = "cpu", **opts) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError("sherpa-onnx backend needs: pip install sherpa-onnx") from e
        key = (os.path.abspath(model), tuple(sorted((k, repr(v)) for k, v in opts.items())))
        with _LOCK:
            rec = _RECOGNIZERS.get(key)
            if rec is None:
                rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=os.path.join(model, opts.get("tokens", "tokens.txt")),
                    encoder=_pick(model, "encoder", opts.get("encoder")),
                    decoder=_pick(model, "decoder", opts.get("decoder")),
                    joiner=_pick(model, "joiner", opts.get("joiner")),
                    num_threads=int(opts.get("num_threads", 2)),
                    sample_rate=self._sample_rate,
                    feature_dim=80,
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=float(opts.get("rule1", 2.4)),
                    rule2_min_trailing_silence=float(opts.get("rule2", 1.2)),
                    rule3_min_utterance_length=float(opts.get("rule3", 20.0)),
                    decoding_method=opts.get("decoding_method", "greedy_search"),
                    provider=opts.get("provider", "cpu"),
                )
                _RECOGNIZERS[key] = rec
        self._recognizer = rec
        self._stream = rec.create_stream()

    def accept(self, chunk: np.ndarray) -> None:
        self._stream.accept_waveform(self._sample_rate, np.asarray(chunk, dtype=np.float32))
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _text(self) -> str:
        result = self._recognizer.get_result(self._stream)
        return (result if isinstance(result, str) else getattr(result, "text", "")).strip()

    def partial(self) -> Optional[Segment]:
        text = self._text()
        return Segment(text=text) if text else None

    def is_endpoint(self) -> bool:
        return bool(self._recognizer.is_endpoint(self._stream))

    def reset(self) -> None:
        self._recognizer.reset(self._stream)
