from concordantmodes.options import Options
import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag

options_kwargs = {
    "cluster": "slurm",
    "program_b": "molpro",
    "program_a": "molpro",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_b": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_b": 9,
    "cart_insert_a": 9,
    "man_proj": True,
    "coords": "Custom",
    "success_regex_b": r"Molpro calculation terminated",
    "success_regex_a": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)


def normalize(mat):
    return 1 / norm(mat, axis=0) * mat


unc = np.eye(1)

CH_str = normalize(
    np.array(
        [
            [1, 1, 1],
            [2, -1, -1],
            [0, 1, -1],
        ]
    ).T
)

CH_ang = normalize(
    np.array(
        [
            [1, 1, 1, -1, -1, -1],
            [2, -1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 2, -1, -1],
            [0, 0, 0, 0, 1, -1],
        ]
    ).T
)

tor = normalize(
    np.array(
        [
            [1, 1, 1],
        ]
    ).T
)

Proj = block_diag(unc, CH_str, unc, unc, CH_ang, tor)

sym_sort = [[0, 1, 2, 4, 5, 6, 7, 9], [3, 8, 10, 11]]

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj, proj=Proj)
CMA_obj.run(sym_sort=sym_sort)
