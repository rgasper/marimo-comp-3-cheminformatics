"""Project-wide exceptions."""


class FingerprintsError(Exception):
    """Base exception."""


class FingerprintGenerationError(FingerprintsError):
    """Raised when a fingerprint cannot be generated for a molecule."""


class DataLoadError(FingerprintsError):
    """Raised when data loading fails."""


class ModelLoadError(FingerprintsError):
    """Raised when a neural model cannot be loaded."""
