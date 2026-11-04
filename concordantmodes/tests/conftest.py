"""The tests read reference data through paths relative to this directory
(at import time), so run them from here wherever pytest was started."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    os.chdir(HERE)
