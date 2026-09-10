"""Exceptions raised by the registry."""


class RegistryError(Exception):
    """Base class for all registry-related errors."""


class _PlainStrKeyError(KeyError):
    """KeyError variant that str()s as the plain message.

    `KeyError.__str__` reprs its args (so `str(KeyError("foo"))` is `"'foo'"`),
    which makes CLI/user-facing messages ugly. Subclasses still behave like a
    KeyError for `except` purposes; callers just get clean text from `str(e)`.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class UnknownMitigationError(_PlainStrKeyError):
    """Raised when a mitigation name is not in the merged registry."""


class UnknownEnvironmentError(_PlainStrKeyError):
    """Raised when an environment name is not in the merged registry."""


class UnknownAgentError(_PlainStrKeyError):
    """Raised when an agent name is not in the merged registry."""


class RegistryCollisionError(RegistryError):
    """Raised when two contributors register the same name in any registry."""
