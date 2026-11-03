class UserFacingError(Exception):
    """Contains only a safe message suitable for a Telegram reply."""


class ProviderError(UserFacingError):
    """Sanitized upstream failure; never includes request/response bodies."""


class RecognitionError(ProviderError):
    """Recognition failed: log metadata, but do not send a chat reply."""


class NoSpeechError(RecognitionError):
    """A completed recognition request returned no speech."""
