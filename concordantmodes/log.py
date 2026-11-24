"""Logging setup for Concordant Modes.

Every module logs through ``logging.getLogger(__name__)``, so all messages
belong to the ``concordantmodes`` logger. Nothing is printed until
:func:`setup_logging` is called (``ConcordantModes.run`` and the ``cma``
command do this), which keeps library use and the test suite quiet.

Console detail follows the ``verbosity`` option:

* ``"quiet"``  - warnings and errors only
* ``"normal"`` - progress and the final results (default)
* ``"debug"``  - everything, including intermediate matrices

The full debug-level record of a run is always written to ``cma.log``
(the ``log_file`` option; set it to ``""`` to disable).
"""

import logging
import sys

LOGGER_NAME = "concordantmodes"
LEVELS = {"quiet": logging.WARNING, "normal": logging.INFO, "debug": logging.DEBUG}

_HANDLER_TAG = "_cma_handler"


def setup_logging(verbosity="normal", log_file="cma.log"):
    """Attach console (and optional file) handlers to the package logger."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LEVELS[verbosity.lower()])
    console.setFormatter(logging.Formatter("%(message)s"))
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)
    return logger


def debug_print(logger, *args, sep=" "):
    """``print``-style helper: join the arguments and log them at DEBUG."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(sep.join(str(a) for a in args))


def info_print(logger, *args, sep=" "):
    """``print``-style helper: join the arguments and log them at INFO."""
    if logger.isEnabledFor(logging.INFO):
        logger.info(sep.join(str(a) for a in args))
