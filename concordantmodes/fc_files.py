"""Where force-constant and gradient files live.

Every force-constant (``fc_*.dat``) and gradient (``fc_*.grad``) file CMA
writes goes into the ``force_constants/`` folder of the run directory.
When reading, CMA looks there first and then in the run directory itself,
so runs started with older versions (and CMA-0B Hessians placed next to
the zmat file) keep working.
"""

import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

FC_DIR = "force_constants"

_FLOAT = re.compile(r"-?\d+\.\d+(?:[eEdD][-+]?\d+)?")


def fc_name(level, cartesian=False, ext="dat"):
    """``fc_int_a.dat``, ``fc_cart_b.grad``, ..."""
    return "fc_{}_{}.{}".format("cart" if cartesian else "int", level.lower(), ext)


def output_path(rootdir, name):
    """Path a file is written to (creates force_constants/ if needed)."""
    folder = os.path.join(rootdir, FC_DIR)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def find(rootdir, name, quiet=False):
    """Existing path of ``name`` (force_constants/ first, then the run
    directory), or None."""
    for path in (os.path.join(rootdir, FC_DIR, name), os.path.join(rootdir, name)):
        if os.path.exists(path):
            if not quiet and os.path.dirname(path) == os.path.normpath(rootdir):
                logger.info(
                    "Reading %s from the run directory; new files are written to %s/.",
                    name,
                    FC_DIR,
                )
            return path
    return None


def display(name):
    """How a file is named in messages: force_constants/<name>."""
    return "{}/{}".format(FC_DIR, name)


def write_vector(path, vector):
    """Write a gradient vector, three values per line."""
    values = np.asarray(vector, dtype=float).flatten()
    lines = []
    for start in range(0, len(values), 3):
        lines.append("".join("{:20.10f}".format(v) for v in values[start : start + 3]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_vector(path):
    with open(path) as fh:
        text = fh.read().replace("D", "E").replace("d", "e")
    return np.array([float(x) for x in _FLOAT.findall(text)])
