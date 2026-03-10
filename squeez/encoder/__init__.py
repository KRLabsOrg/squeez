"""Encoder-based line classifier for tool output extraction."""

__all__ = ["SqueezEncoderConfig", "SqueezEncoderForLineClassification"]


def __getattr__(name: str):
    """Lazily import encoder model classes so lightweight helpers stay optional."""
    if name in __all__:
        from squeez.encoder.model import SqueezEncoderConfig, SqueezEncoderForLineClassification

        return {
            "SqueezEncoderConfig": SqueezEncoderConfig,
            "SqueezEncoderForLineClassification": SqueezEncoderForLineClassification,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
