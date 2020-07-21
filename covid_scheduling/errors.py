"""Scheduling errors."""


class AssignmentError(Exception):
    """Raised for errors when generating assignments."""
    def __init__(self, message: str):
        self.message = message
