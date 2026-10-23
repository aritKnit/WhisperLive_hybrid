"""External YAML configuration.

Layers, lowest to highest priority:

1. ``config/engine.default.yaml``   every tunable with the value the code used to hardcode
2. a profile (``--profile sensevoice`` -> ``config/engine.sensevoice.yaml``) or ``--config FILE``
3. environment variables ``ENGINE__SECTION__KEY=value``
4. command-line flags (applied by ``run_server.py``)

Overlays may only set keys that exist in the defaults, so a typo is an error, not a silent no-op.
``backend.options`` is free-form because plug-in backends define their own keys.
"""
import ast
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(os.environ.get("ENGINE_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))
DEFAULT_FILE = CONFIG_DIR / "engine.default.yaml"
ENV_PREFIX = "ENGINE__"
FREE_FORM = {("backend", "options")}


class ConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any], path=(), source="overlay") -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        here = path + (key,)
        if key not in out and path not in FREE_FORM:
            raise ConfigError(f"Unknown config key '{'.'.join(here)}' in {source}")
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value, here, source)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _env_overlay(env) -> Dict[str, Any]:
    overlay: Dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = {"true": True, "false": False, "null": None, "none": None}.get(raw.strip().lower(), raw)
        node = overlay
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overlay


class Config:
    def __init__(self, data: Dict[str, Any], defaults: Dict[str, Any], sources):
        self.data = data
        self.defaults = defaults
        self.sources = sources

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def changed(self, section: str) -> Dict[str, Any]:
        """Keys of ``section`` whose value differs from the shipped default."""
        now, base = self.get(section, {}), self._default(section)
        return {k: v for k, v in now.items() if base.get(k) != v}

    def _default(self, section: str) -> Dict[str, Any]:
        node: Any = self.defaults
        for part in section.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        return node


def load_config(profile: Optional[str] = None, path: Optional[str] = None, env=None) -> Config:
    """Load defaults, then a profile or explicit file, then ``ENGINE__*`` environment overrides."""
    env = os.environ if env is None else env
    if not DEFAULT_FILE.exists():
        raise ConfigError(
            f"Default config not found at {DEFAULT_FILE}. Run from the repository root, copy the "
            f"config/ directory next to run_server.py, or set ENGINE_CONFIG_DIR."
        )
    defaults = _read(DEFAULT_FILE)
    data = copy.deepcopy(defaults)
    sources = [str(DEFAULT_FILE)]

    overlay_file = None
    if path:
        overlay_file = Path(path)
    elif profile:
        overlay_file = CONFIG_DIR / f"engine.{profile}.yaml"
    if overlay_file is not None:
        if not overlay_file.exists():
            raise ConfigError(f"Config file not found: {overlay_file}")
        data = _deep_merge(data, _read(overlay_file), source=str(overlay_file))
        sources.append(str(overlay_file))

    env_overlay = _env_overlay(env)
    if env_overlay:
        data = _deep_merge(data, env_overlay, source="environment")
        sources.append("environment")

    cfg = Config(data, defaults, sources)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.get("audio.buffer_trim_s") >= cfg.get("audio.max_buffer_s"):
        raise ConfigError("audio.buffer_trim_s must be smaller than audio.max_buffer_s")
    if cfg.get("server.max_clients") < 1:
        raise ConfigError("server.max_clients must be >= 1")
    if cfg.get("server.max_connection_time_s") <= 0:
        raise ConfigError("server.max_connection_time_s must be > 0")
    if cfg.get("audio.sample_rate") != 16000:
        raise ConfigError("audio.sample_rate other than 16000 is not supported by the bundled VAD and models")


# dotted config key -> ServeClientBase class attribute
_BASE_CONSTANTS = {
    "audio.max_buffer_s": "MAX_BUFFER_DURATION_S",
    "audio.buffer_trim_s": "BUFFER_TRIM_DURATION_S",
    "audio.clip_threshold_s": "CLIP_THRESHOLD_DURATION_S",
    "audio.clip_tail_s": "CLIP_TAIL_DURATION_S",
    "stream.first_frame_wait_timeout_s": "FIRST_FRAME_WAIT_TIMEOUT_S",
    "stream.audio_wait_timeout_s": "AUDIO_WAIT_TIMEOUT_S",
    "stream.voice_wait_timeout_s": "VOICE_WAIT_TIMEOUT_S",
    "stream.error_backoff_s": "ERROR_BACKOFF_S",
    "stream.max_transcript_length": "MAX_TRANSCRIPT_LENGTH",
    "stream.max_translation_queue": "MAX_TRANSLATION_QUEUE_SIZE",
}


def apply_runtime_constants(cfg: Config) -> Dict[str, Any]:
    """Push buffer and timing values into ``ServeClientBase`` (shared by every backend)."""
    from whisper_live.backend.base import ServeClientBase
    applied = {}
    for key, attr in _BASE_CONSTANTS.items():
        value = cfg.get(key)
        if value is not None:
            setattr(ServeClientBase, attr, value)
            applied[attr] = value
    return applied


# decoding keys that are per-session client options, not model-wide knobs
_PER_SESSION_DECODING = {"task", "language", "initial_prompt", "hotwords", "word_timestamps"}


def decoding_overrides(cfg: Config) -> Dict[str, Any]:
    """Model-wide decoding parameters the operator changed from the library defaults."""
    return {k: v for k, v in cfg.changed("decoding").items() if k not in _PER_SESSION_DECODING}


_LEGACY_BACKENDS = ("faster_whisper", "tensorrt", "openvino")


def backend_options(cfg: Config) -> Dict[str, Any]:
    """Options for ``backend.load()`` and the engine sessions, from the config file.

    Empty for the legacy Whisper backends, which take their settings from the older flags.
    """
    backend = cfg.get("backend.type")
    if backend in _LEGACY_BACKENDS:
        return {}
    opts: Dict[str, Any] = {k: v for k, v in (cfg.get("backend.engine") or {}).items() if v is not None}
    specific = cfg.get(f"backend.{backend}")
    if isinstance(specific, dict):
        opts.update({k: v for k, v in specific.items() if v is not None})
    opts.update(cfg.get("backend.options") or {})
    return opts


# argparse dest -> (dotted config key, transform)
_CLI_DEFAULTS = {
    "port": ("server.port", None),
    "rest_port": ("server.rest_port", None),
    "enable_rest": ("server.enable_rest", None),
    "metrics_port": ("server.metrics_port", None),
    "cors_origins": ("server.cors_origins", None),
    "api_key": ("server.api_key", None),
    "rate_limit_rpm": ("server.rate_limit_rpm", None),
    "max_clients": ("server.max_clients", None),
    "max_connection_time": ("server.max_connection_time_s", None),
    "no_single_model": ("server.single_model", lambda v: not v),
    "raw_pcm_input": ("server.raw_pcm_input", None),
    "cache_path": ("server.cache_path", None),
    "omp_num_threads": ("server.omp_num_threads", None),
    "backend": ("backend.type", None),
    "trt_model_path": ("backend.tensorrt.model_path", None),
    "trt_multilingual": ("backend.tensorrt.multilingual", None),
    "trt_py_session": ("backend.tensorrt.py_session", None),
    "model": ("model.active", None),
    "default_model": ("model.default_name", None),
    "faster_whisper_custom_model_path": ("model.custom_path", None),
    "batch_inference": ("batch.enabled", None),
    "batch_max_size": ("batch.max_size", None),
    "batch_window_ms": ("batch.window_ms", None),
}


def cli_defaults(cfg: Config) -> Dict[str, Any]:
    """Values for ``argparse.set_defaults`` so the YAML supplies defaults and flags still win."""
    out: Dict[str, Any] = {}
    for dest, (key, transform) in _CLI_DEFAULTS.items():
        value = cfg.get(key)
        if value is None:
            continue
        out[dest] = transform(value) if transform else value
    return out
