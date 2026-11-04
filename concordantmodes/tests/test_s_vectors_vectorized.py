"""
Check that the vectorized B-matrix construction (SVectors._build_B)
reproduces the original per-coordinate loop (SVectors._build_B_reference)
for every coordinate type, centroid coordinates, and displaced geometries.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from concordantmodes.options import Options
from concordantmodes.s_vectors import SVectors
from concordantmodes.zmat import Zmat

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_data")

CASES = [
    # stretch, bend, torsion, out-of-plane, L, Lx and Ly coordinates
    ("zmat_test", "zmat_custom", "Custom"),
    ("s_vec_test", "zmat", "Custom"),
    ("s_vec_test", "zmat", "Delocalized"),
    ("symmetry_test", "c3v_ammonia_zmat_custom", "Custom"),
    # centroid (";"-grouped) bond, angle and torsion
    ("s_vec_centroid_test", "zmat", "Custom"),
]


def _load(directory, zmat_name, coords):
    cwd = os.getcwd()
    os.chdir(os.path.join(REF, directory))
    try:
        options = Options()
        options.coords = coords
        with contextlib.redirect_stdout(io.StringIO()):
            zmat = Zmat(options)
            zmat.run(zmat_name=zmat_name)
    finally:
        os.chdir(cwd)
    return zmat, options


@pytest.mark.parametrize("directory, zmat_name, coords", CASES)
def test_vectorized_B_matches_reference(directory, zmat_name, coords):
    zmat, options = _load(directory, zmat_name, coords)
    s_vec = SVectors(zmat, options)
    x0 = np.array(zmat.cartesians_b, dtype=float)
    rng = np.random.default_rng(1234)
    for trial in range(5):
        shift = 0 if trial == 0 else rng.normal(scale=0.05, size=x0.shape)
        s_vec.carts = x0 + shift
        B_ref = s_vec._build_B_reference().astype(float)
        B_vec = s_vec._build_B()
        assert B_vec.shape == B_ref.shape
        assert np.allclose(B_vec, B_ref, rtol=0, atol=1e-12)


def test_run_uses_vectorized_B():
    zmat, options = _load("zmat_test", "zmat_custom", "Custom")
    s_vec = SVectors(zmat, options)
    with contextlib.redirect_stdout(io.StringIO()):
        s_vec.run(zmat.cartesians_b, True)
    B_ref = s_vec._build_B_reference().astype(float)
    assert np.allclose(s_vec.B, B_ref, rtol=0, atol=1e-12)
    # The per-coordinate s-vector dictionaries are still populated.
    assert len(s_vec.s_2center_dict) == len(zmat.bond_indices)
