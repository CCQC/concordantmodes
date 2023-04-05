from functools import reduce
import pytest
import numpy as np
import os
import shutil
import re

from concordantmodes.int2cart import Int2Cart
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.options import Options
from concordantmodes.zmat import Zmat

coord1 = "Redundant"
coord2 = "ZMAT"
coord3 = "Custom"
file1 = "zmat_red"
file2 = "zmat_zmat"
file3 = "zmat_custom"
# ZMAT read data
ref_Red = ["1 2\n", "1 3\n", "1 4\n", "1 5\n", "2 6\n"]
ref_ZMAT = ["C\n", "O 1\n", "H 1 2\n", "H 1 2 3\n", "H 1 2 4\n", "H 2 1 3\n"]
ref_Custom = [
    "1 2\n",
    "1 3\n",
    "1 4\n",
    "1 5\n",
    "2 6\n",
    "2 1 3\n",
    "2 1 4\n",
    "2 1 5\n",
    "3 1 4\n",
    "4 1 5\n",
    "5 1 3\n",
    "6 2 1\n",
    "6 2 1 4 T\n",
    "4 1 3 5 O\n",
    "3 1 2 6 Lx\n",
    "3 1 2 6 Ly\n",
    "5 1 2 4 L\n",
]
zmat_read = [
    (coord1, ref_Red, file1),
    (coord2, ref_ZMAT, file2),
    (coord2, ref_Custom, file3),
]


def make_zmat(type: str, option: str):
    os.chdir("./ref_data/zmat_test/")
    options = Options()

    options.coords = option
    ZMAT = Zmat(options)
    output_test_red = ZMAT.zmat_read(type)
    ZMAT.zmat_process(output_test_red)
    os.chdir("../../")
    return ZMAT


zmat = make_zmat("zmat_zmat", "ZMAT")
redundant_zmat = make_zmat("zmat_red", "Redundant")
custom_zmat = make_zmat("zmat_custom", "Custom")

ref_bond_indices = [
    (zmat, [["2", "1"], ["3", "1"], ["4", "1"], ["5", "1"], ["6", "2"]]),
    (redundant_zmat, [["1", "2"], ["1", "3"], ["1", "4"], ["1", "5"], ["2", "6"]]),
    (custom_zmat, [("1", "2"), ("1", "3"), ("1", "4"), ("1", "5"), ("2", "6")]),
]

ref_bond_variables = [
    (zmat, ["R1", "R2", "R3", "R4", "R5"]),
    (redundant_zmat, ["R1", "R2", "R3", "R4", "R5"]),
    (custom_zmat, ["R1", "R2", "R3", "R4", "R5"]),
]

ref_angle_indices = [
    (zmat, [["3", "1", "2"], ["4", "1", "2"], ["5", "1", "2"], ["6", "2", "1"]]),
    (
        redundant_zmat,
        [
            ["2", "1", "3"],
            ["2", "1", "4"],
            ["2", "1", "5"],
            ["1", "2", "6"],
            ["3", "1", "4"],
            ["3", "1", "5"],
            ["4", "1", "5"],
        ],
    ),
    (
        custom_zmat,
        [
            ("2", "1", "3"),
            ("2", "1", "4"),
            ("2", "1", "5"),
            ("3", "1", "4"),
            ("4", "1", "5"),
            ("5", "1", "3"),
            ("6", "2", "1"),
        ],
    ),
]

ref_angle_variables = [
    (zmat, ["A2", "A3", "A4", "A5"]),
    (redundant_zmat, ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]),
    (custom_zmat, ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]),
]

ref_tors_indices = [
    (zmat, [["4", "1", "2", "3"], ["5", "1", "2", "4"], ["6", "2", "1", "3"]]),
    (
        redundant_zmat,
        [
            ["3", "1", "2", "4"],
            ["3", "1", "2", "5"],
            ["3", "1", "2", "6"],
            ["2", "1", "3", "4"],
            ["2", "1", "3", "5"],
            ["4", "1", "2", "5"],
            ["4", "1", "2", "6"],
            ["2", "1", "4", "3"],
            ["2", "1", "4", "5"],
            ["5", "1", "2", "6"],
            ["2", "1", "5", "3"],
            ["2", "1", "5", "4"],
            ["4", "1", "3", "5"],
            ["3", "1", "4", "5"],
            ["3", "1", "5", "4"],
        ],
    ),
    (custom_zmat, [("6", "2", "1", "4")]),
]

ref_tors_variables = [
    (zmat, ["D3", "D4", "D5"]),
    (
        redundant_zmat,
        [
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "D7",
            "D8",
            "D9",
            "D10",
            "D11",
            "D12",
            "D13",
            "D14",
            "D15",
        ],
    ),
    (custom_zmat, ["D1"]),
]

ref_oop_indices = (custom_zmat.oop_indices, [("4", "1", "3", "5")])
ref_lin_indices = (custom_zmat.lin_indices, [("5", "1", "2", "4")])
ref_linx_indices = (custom_zmat.linx_indices, [("3", "1", "2", "6")])
ref_liny_indices = (custom_zmat.liny_indices, [("3", "1", "2", "6")])
ref_oop_variables = (custom_zmat.oop_variables, ["O1"])
ref_lin_variables = (custom_zmat.lin_variables, ["L1"])
ref_linx_variables = (custom_zmat.linx_variables, ["Lx1"])
ref_liny_variables = (custom_zmat.liny_variables, ["Ly1"])


@pytest.mark.parametrize("option, expected, file_name", zmat_read)
def test_zmat_read(option, expected, file_name):
    os.chdir("./ref_data/zmat_test/")
    options = Options()

    options.coords = option
    ZMAT = Zmat(options)

    output_test = ZMAT.zmat_read(file_name)

    os.chdir("../../")
    assert expected == output_test


@pytest.mark.parametrize(
    "ZMAT, ref_bond_indices",
    ref_bond_indices,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_bond_indices(ZMAT, ref_bond_indices):
    if isinstance(ZMAT.bond_indices, np.ndarray):
        ZMAT.bond_indices = ZMAT.bond_indices.tolist()
    assert ZMAT.bond_indices == ref_bond_indices


@pytest.mark.parametrize(
    "ZMAT, ref_bond_variables",
    ref_bond_variables,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_bond_variables(ZMAT, ref_bond_variables):
    assert list(ZMAT.bond_variables) == ref_bond_variables


@pytest.mark.parametrize(
    "ZMAT, ref_angle_indices",
    ref_angle_indices,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_angle_indices(ZMAT, ref_angle_indices):
    if isinstance(ZMAT.angle_indices, np.ndarray):
        ZMAT.angle_indices = ZMAT.angle_indices.tolist()
    assert list(ZMAT.angle_indices) == ref_angle_indices


@pytest.mark.parametrize(
    "ZMAT, ref_angle_variables",
    ref_angle_variables,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_angle_variables(ZMAT, ref_angle_variables):
    assert list(ZMAT.angle_variables) == ref_angle_variables


@pytest.mark.parametrize(
    "ZMAT, ref_tors_indices",
    ref_tors_indices,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_torsion_indices(ZMAT, ref_tors_indices):
    if isinstance(ZMAT.torsion_indices, np.ndarray):
        ZMAT.torsion_indices = ZMAT.torsion_indices.tolist()
    print(ZMAT.torsion_indices)
    print(ref_tors_indices)
    assert list(ZMAT.torsion_indices) == ref_tors_indices


@pytest.mark.parametrize(
    "ZMAT, ref_tors_variables",
    ref_tors_variables,
    ids=["standard zmat", "automatic redundant", "custom"],
)
def test_zmat_torsion_variables(ZMAT, ref_tors_variables):
    assert list(ZMAT.torsion_variables) == ref_tors_variables


@pytest.mark.parametrize(
    "custom_zmat_coords, reference_coords",
    [
        ref_oop_indices,
        ref_oop_variables,
        ref_lin_indices,
        ref_lin_variables,
        ref_linx_indices,
        ref_linx_variables,
        ref_liny_indices,
        ref_liny_variables,
    ],
    ids=[
        "redundant out of plane indices",
        "redundant out of place variables",
        "redundant lin indices",
        "redundant lin variables",
        "redundant linx indices",
        "redundant linx variables",
        "redundant_liny indices",
        "redundant_liny variables",
    ],
)
def test_custom_zmat(custom_zmat_coords, reference_coords):
    assert custom_zmat_coords == reference_coords


# Only need to test the Custom internal coordinates
# TODO pythonize
def test_zmat_calc():
    os.chdir("./ref_data/zmat_test/")
    options = Options()
    errors = []

    options.coords = "Custom"
    ZMAT = Zmat(options)
    output_test_red = ZMAT.zmat_read("zmat_custom")
    ZMAT.zmat_process(output_test_red)

    ZMAT.zmat_calc()

    var_dict_ref = {
        "R1": 2.685006407296404,
        "R2": 2.0576342503795035,
        "R3": 2.068739344106188,
        "R4": 2.068739340043222,
        "R5": 1.8130883453288267,
        "A1": 106.8765896566676,
        "A2": 112.2824413394013,
        "A3": 112.28244150454864,
        "A4": 108.22376477570377,
        "A5": 108.7851835780631,
        "A6": 108.22360552816234,
        "A7": 107.41157765289944,
        "D1": 61.480653653479756,
        "O1": 57.2185581514689,
        "L1": -45.926274971749756,
        "Lx1": -54.67048011096265,
        "Ly1": -0.0022793163188044303,
    }
    var_dict_custom = ZMAT.variable_dictionary_final

    if np.setdiff1d(var_dict_ref, var_dict_custom).size:
        errors.append("Custom variables do not match.")

    os.chdir("../../")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


# Only need to test the Custom internal coordinates
def test_zmat_compile():
    os.chdir("./ref_data/zmat_test/")
    options = Options()
    errors = []

    options.coords = "Custom"
    ZMAT = Zmat(options)
    output_test_custom = ZMAT.zmat_read("zmat_custom")
    ZMAT.zmat_process(output_test_custom)

    ZMAT.zmat_calc()

    ZMAT.zmat_compile()

    print(ZMAT.index_dictionary)

    index_dict_ref = {
        "R1": ("1", "2"),
        "R2": ("1", "3"),
        "R3": ("1", "4"),
        "R4": ("1", "5"),
        "R5": ("2", "6"),
        "A1": ("2", "1", "3"),
        "A2": ("2", "1", "4"),
        "A3": ("2", "1", "5"),
        "A4": ("3", "1", "4"),
        "A5": ("4", "1", "5"),
        "A6": ("5", "1", "3"),
        "A7": ("6", "2", "1"),
        "D1": ("6", "2", "1", "4"),
        "O1": ("4", "1", "3", "5"),
        "L1": ("5", "1", "2", "4"),
        "Lx1": ("3", "1", "2", "6"),
        "Ly1": ("3", "1", "2", "6"),
    }
    index_dict_custom = ZMAT.index_dictionary

    if np.setdiff1d(index_dict_ref, index_dict_custom).size:
        errors.append("Custom indices do not match.")

    os.chdir("../../")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

