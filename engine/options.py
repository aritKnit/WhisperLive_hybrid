"""Parsing and negotiation of generic backend options."""
import ast
from typing import Any, Dict, Iterable, List, Optional, Tuple

from whisper_live.engine.types import Capabilities


def parse_backend_opts(pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    """``["beam=5", "hub=hf", "use_itn=true"]`` -> ``{"beam": 5, "hub": "hf", "use_itn": True}``."""
    opts: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--backend_opt expects key=value, got '{pair}'")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--backend_opt has an empty key in '{pair}'")
        low = raw.strip().lower()
        if low in ("true", "false"):
            opts[key] = low == "true"
        elif low in ("none", "null"):
            opts[key] = None
        else:
            try:
                opts[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                opts[key] = raw
    return opts


def negotiate_options(options: Dict[str, Any], caps: Capabilities) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """Reconcile a client's handshake with what the backend can do.

    Returns ``(options, warnings, error)``. ``error`` is set when the request
    cannot be served at all; otherwise unsupported extras are dropped or
    rerouted and each change is reported in ``warnings``.
    """
    options = dict(options)
    warnings: List[str] = []

    language = options.get("language")
    if not caps.supports_language(language):
        return options, warnings, (
            f"Language '{language}' is not supported by this server's model. "
            f"Supported: {', '.join(caps.languages or [])}"
        )

    if options.get("task") == "translate" and not caps.translate:
        options["task"] = "transcribe"
        if not options.get("enable_translation"):
            options["enable_translation"] = True
            options.setdefault("target_language", "en")
        warnings.append(
            "This model cannot translate natively; transcribing and translating to "
            f"'{options.get('target_language', 'en')}' with the separate translation stage instead."
        )

    for key, supported, label in (
        ("hotwords", caps.hotwords, "hotwords"),
        ("initial_prompt", caps.initial_prompt, "initial_prompt"),
        ("word_timestamps", caps.word_timestamps, "word_timestamps"),
    ):
        if options.get(key) and not supported:
            options[key] = None if key != "word_timestamps" else False
            warnings.append(f"'{label}' is not supported by this model and was ignored.")

    return options, warnings, None
