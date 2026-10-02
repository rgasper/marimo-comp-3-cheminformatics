"""Project-wide exceptions."""


class FingerprintsError(Exception):
    """Base exception."""


class DataLoadError(FingerprintsError):
    """Raised when data loading fails."""
