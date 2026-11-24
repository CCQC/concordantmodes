"""Exceptions raised by Concordant Modes.

``CMAError`` covers anything the user has to fix (bad option, missing file,
failed calculation). ``CMAWaiting`` is not an error: it stops the run
cleanly when displacement calculations still have to be run, so the next
``cma run`` (or ``python main.py``) picks up where this one stopped.
"""


class CMAError(RuntimeError):
    """A problem with the user's input, files or calculations."""


class OptionsError(CMAError, ValueError):
    """An unknown option name or an invalid option value."""


class CMAWaiting(Exception):
    """Displacement calculations are not finished yet; rerun later."""

    def __init__(self, message, level=None, missing=(), failed=()):
        super().__init__(message)
        self.level = level
        self.missing = list(missing)
        self.failed = list(failed)
