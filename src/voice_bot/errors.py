class UserFacingError(Exception):
    """Contains only a safe message suitable for a Telegram reply."""


class ProviderError(UserFacingError):
    """Sanitized upstream failure; never includes request/response bodies."""
