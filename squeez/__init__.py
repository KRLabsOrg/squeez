"""Squeez — Squeeze verbose LLM agent tool output down to only the relevant lines."""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy-load encoder classes to avoid importing torch at package level."""
    _encoder_exports = {
        "SqueezEncoderConfig": "SqueezEncoderConfig",
        "SqueezEncoderForLineClassification": "SqueezEncoderForLineClassification",
    }
    if name in _encoder_exports:
        import importlib

        mod = importlib.import_module("squeez.encoder")
        return getattr(mod, name)
    raise AttributeError(f"module 'squeez' has no attribute {name!r}")
