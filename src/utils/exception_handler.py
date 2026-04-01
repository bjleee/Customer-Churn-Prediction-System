"""Utilities for building concise, structured exception messages."""

import traceback

_MAX_ERROR_MESSAGE_LENGTH = 180


def _truncate_message(message: str, max_length: int = _MAX_ERROR_MESSAGE_LENGTH) -> str:
    """Truncates long messages for compact logs.

    Args:
        message: Original error message.
        max_length: Maximum allowed message length.

    Returns:
        Truncated message with trailing ellipsis when needed.
    """
    if len(message) <= max_length:
        return message

    return f"{message[:max_length]}..."


def error_message_detail(raw_error_message: Exception) -> str:
    """Builds a concise error string with source location and message.

    Args:
        raw_error_message: The caught exception object.

    Returns:
        Formatted one-line message suitable for logging.
    """
    trace = traceback.extract_tb(raw_error_message.__traceback__)
    last_frame = trace[-1] if trace else None

    file_name = last_frame.filename if last_frame else "unknown_file"
    line_number = last_frame.lineno if last_frame else -1

    short_message = _truncate_message(str(raw_error_message))
    return (
        f"Error occurred in python file [{file_name}] at line "
        f"[{line_number}] | message: {short_message}"
    )


class CustomError(Exception):
    """Project-specific exception wrapper with compact log-friendly detail."""

    def __init__(self, error: Exception):
        """Initializes the custom exception from an original exception.

        Args:
            error: Original exception that was raised.
        """
        self.error_message = error_message_detail(error)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        """Returns the formatted exception message."""
        return self.error_message
