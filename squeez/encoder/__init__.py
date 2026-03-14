"""Encoder-based line classifier for tool output extraction."""

__all__ = [
    "SqueezEncoderConfig",
    "SqueezEncoderForLineClassification",
    "PooledLineConfig",
    "PooledLineClassifier",
]


def __getattr__(name: str):
    """Lazily import encoder model classes so lightweight helpers stay optional."""
    if name in ("SqueezEncoderConfig", "SqueezEncoderForLineClassification"):
        from squeez.encoder.model import SqueezEncoderConfig, SqueezEncoderForLineClassification

        return {
            "SqueezEncoderConfig": SqueezEncoderConfig,
            "SqueezEncoderForLineClassification": SqueezEncoderForLineClassification,
        }[name]
    if name in ("PooledLineConfig", "PooledLineClassifier"):
        from squeez.encoder.sentence import PooledLineClassifier, PooledLineConfig

        return {
            "PooledLineConfig": PooledLineConfig,
            "PooledLineClassifier": PooledLineClassifier,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
