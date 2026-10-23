"""Backend registry with lazy imports.

``name -> "module:Class"``. Nothing is imported until a backend is actually
used, so a SenseVoice or sherpa-onnx deployment never imports faster-whisper,
ctranslate2 or TensorRT. ``--backend my_pkg.mod:MyBackend`` loads an
out-of-tree backend with no registration.
"""
import importlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from whisper_live.engine.types import Capabilities

SESSION = "session"      # legacy: model and session are one class (the three Whisper backends)
OFFLINE = "offline"      # OfflineBackend driven by WindowedSession
STREAMING = "streaming"  # StreamingBackend driven by StreamingSession

_WHISPER_CAPS = Capabilities(
    translate=True, hotwords=True, initial_prompt=True,
    word_timestamps=True, language_detection=True,
)
_LIMITED_WHISPER_CAPS = Capabilities(translate=True, language_detection=True)


@dataclass(frozen=True)
class BackendSpec:
    name: str
    target: str                          # "module:Class"
    kind: Optional[str] = None           # None = decided from the class once imported
    capabilities: Optional[Capabilities] = None   # static, so the handshake needs no import


_REGISTRY: Dict[str, BackendSpec] = {}


def register_backend(name: str, target: str, kind: Optional[str] = None,
                     capabilities: Optional[Capabilities] = None) -> BackendSpec:
    if ":" not in target:
        raise ValueError(f"backend target must be 'module:Class', got '{target}'")
    spec = BackendSpec(name, target, kind, capabilities)
    _REGISTRY[name] = spec
    return spec


register_backend("faster_whisper", "whisper_live.backend.faster_whisper_backend:ServeClientFasterWhisper",
                 SESSION, _WHISPER_CAPS)
register_backend("tensorrt", "whisper_live.backend.trt_backend:ServeClientTensorRT",
                 SESSION, _LIMITED_WHISPER_CAPS)
register_backend("openvino", "whisper_live.backend.openvino_backend:ServeClientOpenVINO",
                 SESSION, _LIMITED_WHISPER_CAPS)
register_backend("sensevoice", "whisper_live.backend.sensevoice_backend:SenseVoiceBackend", OFFLINE)
register_backend("sherpa_onnx", "whisper_live.backend.sherpa_onnx_backend:SherpaOnnxOnlineBackend", STREAMING)


def registered_names():
    return sorted(_REGISTRY)


def is_known(name: str) -> bool:
    return name in _REGISTRY or _looks_like_target(name)


def _looks_like_target(name: str) -> bool:
    mod, sep, cls = name.partition(":")
    return bool(sep and mod and cls)


def resolve(name: str) -> BackendSpec:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if _looks_like_target(name):
        return BackendSpec(name, name)
    raise ValueError(
        f"'{name}' is not a known backend. Choose one of {registered_names()} "
        f"or give 'package.module:ClassName'."
    )


def load_class(spec: BackendSpec):
    module_name, _, class_name = spec.target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def kind_of(spec: BackendSpec) -> str:
    if spec.kind:
        return spec.kind
    from whisper_live.engine.base import OfflineBackend, StreamingBackend
    cls = load_class(spec)
    if isinstance(cls, type) and issubclass(cls, StreamingBackend):
        return STREAMING
    if isinstance(cls, type) and issubclass(cls, OfflineBackend):
        return OFFLINE
    return SESSION


def capabilities_for(name: str) -> Capabilities:
    """Capabilities without loading a model. Imports the class only if none were declared."""
    spec = resolve(name)
    if spec.capabilities is not None:
        return spec.capabilities
    return getattr(load_class(spec), "CAPABILITIES", Capabilities())


def capabilities_payload(name: str) -> Dict[str, Any]:
    return capabilities_for(name).to_dict()


# ---- shared offline instances: one set of weights per (backend, model, opts) -------------

_SHARED: Dict[Tuple, Any] = {}
_SHARED_LOCK = threading.Lock()


def _freeze(opts: Dict[str, Any]) -> Tuple:
    return tuple(sorted((k, repr(v)) for k, v in (opts or {}).items()))


def get_offline_backend(spec: BackendSpec, model: str, device: str = "cpu",
                        opts: Optional[Dict[str, Any]] = None, warmup: bool = False):
    """Return the process-wide loaded instance, creating and loading it once."""
    key = (spec.name, model, device, _freeze(opts or {}))
    with _SHARED_LOCK:
        backend = _SHARED.get(key)
        if backend is None:
            backend = load_class(spec)()
            backend.load(model, device, **(opts or {}))
            if warmup:
                backend.warmup()
            _SHARED[key] = backend
            # lock travels with the instance so every session serialises on the same one
            backend._engine_lock = threading.Lock()
        return backend


def new_streaming_backend(spec: BackendSpec, model: str, device: str = "cpu",
                          opts: Optional[Dict[str, Any]] = None):
    backend = load_class(spec)()
    backend.load(model, device, **(opts or {}))
    return backend


def clear_shared() -> None:
    with _SHARED_LOCK:
        _SHARED.clear()
