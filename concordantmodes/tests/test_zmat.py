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


def conv_array(arr):
    buff_array = arr
    arr = np.array([],dtype=object)
    for ind_set in buff_array:
        arr = np.append(arr,0)
        arr[-1] = np.array(ind_set,dtype=object)
    return arr

coord1 = "Delocalized"
coord2 = "ZMAT"
coord3 = "Custom"
file1 = "zmat_del"
file2 = "zmat_zmat"
file3 = "zmat_custom"
# ZMAT read data
ref_Deloc = ["1 2\n", "1 3\n", "1 4\n", "1 5\n", "2 6\n"]
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
    (coord1, ref_Deloc, file1),
    (coord2, ref_ZMAT, file2),
    (coord2, ref_Custom, file3),
]


def make_zmat(type: str, option: str):
    os.chdir("./ref_data/zmat_test/")
    options = Options()

    options.coords = option
    ZMAT = Zmat(options)
    output_test_del = ZMAT.zmat_read(type)
    ZMAT.zmat_process(output_test_del)
    os.chdir("../../")
    return ZMAT


zmat = make_zmat("zmat_zmat", "ZMAT")
delocalized_zmat = make_zmat("zmat_del", "Delocalized")
custom_zmat = make_zmat("zmat_custom", "Custom")

zmat_buff = [["2", "1"], ["3", "1"], ["4", "1"], ["5", "1"], ["6", "2"]]
zmat_array = conv_array(zmat_buff)

deloc_buff = [["1", "2"], ["1", "3"], ["1", "4"], ["1", "5"], ["2", "6"]]
deloc_array = conv_array(deloc_buff)

custom_buff = [["1", "2"], ["1", "3"], ["1", "4"], ["1", "5"],["2", "6"]]
custom_array = conv_array(custom_buff)

ref_bond_indices = [
    (zmat, zmat_array),
    (delocalized_zmat, deloc_array),
    (custom_zmat, custom_array),
]

ref_bond_variables = [
    (zmat, ["R1", "R2", "R3", "R4", "R5"]),
    (delocalized_zmat, ["R1", "R2", "R3", "R4", "R5"]),
    (custom_zmat, ["R1", "R2", "R3", "R4", "R5"]),
]

zmat_buff = [["3", "1", "2"], ["4", "1", "2"], ["5", "1", "2"], ["6", "2", "1"]]
zmat_array = conv_array(zmat_buff)

deloc_buff = [["2", "1", "3"],
            ["2", "1", "4"],
            ["2", "1", "5"],
            ["1", "2", "6"],
            ["3", "1", "4"],
            ["3", "1", "5"],
            ["4", "1", "5"],]
deloc_array = conv_array(deloc_buff)

custom_buff = [
            ["2", "1", "3"],
            ["2", "1", "4"],
            ["2", "1", "5"],
            ["3", "1", "4"],
            ["4", "1", "5"],
            ["5", "1", "3"],
            ["6", "2", "1"],
        ]
custom_array = conv_array(custom_buff)

ref_angle_indices = [
    (zmat, zmat_array),
    (delocalized_zmat, deloc_array),
    (custom_zmat, custom_array),
]

ref_angle_variables = [
    (zmat, ["A2", "A3", "A4", "A5"]),
    (delocalized_zmat, ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]),
    (custom_zmat, ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]),
]

zmat_buff = [["4", "1", "2", "3"], ["5", "1", "2", "4"], ["6", "2", "1", "3"]]
zmat_array = conv_array(zmat_buff)

deloc_buff =[["6", "2", "1", "3"],
            ["6", "2", "1", "4"],
            ["6", "2", "1", "5"],]
deloc_array = conv_array(deloc_buff)

custom_buff = [["6", "2", "1", "4"]]
custom_array = conv_array(custom_buff)

ref_tors_indices = [
    (zmat, zmat_array),
    (delocalized_zmat, deloc_array),
    (custom_zmat, custom_array),
]

ref_tors_variables = [
    (zmat, ["D3", "D4", "D5"]),
    (
        delocalized_zmat,
        [
            "D1",
            "D2",
            "D3",
        ],
    ),
    (custom_zmat, ["D1"]),
]

ref_oop_indices = (custom_zmat.oop_indices, conv_array([["4", "1", "3", "5"]]))
ref_lin_indices = (custom_zmat.lin_indices, conv_array([["5", "1", "2", "4"]]))
ref_linx_indices = (custom_zmat.linx_indices, conv_array([["3", "1", "2", "6"]]))
ref_liny_indices = (custom_zmat.liny_indices, conv_array([["3", "1", "2", "6"]]))
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
    ids=["standard zmat", "automatic delocalized", "custom"],
)
def test_zmat_bond_indices(ZMAT, ref_bond_indices):
    for i in range(len(ZMAT.bond_indices)):
        for j in range(len(ZMAT.bond_indices[i])):
            assert ZMAT.bond_indices[i][j] == ref_bond_indices[i][j]


@pytest.mark.parametrize(
    "ZMAT, ref_bond_variables",
    ref_bond_variables,
    ids=["standard zmat", "automatic delocalized", "custom"],
)
def test_zmat_bond_variables(ZMAT, ref_bond_variables):
    assert list(ZMAT.bond_variables) == ref_bond_variables


@pytest.mark.parametrize(
    "ZMAT, ref_angle_indices",
    ref_angle_indices,
    ids=["standard zmat", "automatic delocalized", "custom"],
)
def test_zmat_angle_indices(ZMAT, ref_angle_indices):
    for i in range(len(ZMAT.angle_indices)):
        for j in range(len(ZMAT.angle_indices[i])):
            assert ZMAT.angle_indices[i][j] == ref_angle_indices[i][j]


@pytest.mark.parametrize(
    "ZMAT, ref_angle_variables",
    ref_angle_variables,
    ids=["standard zmat", "automatic delocalized", "custom"],
)
def test_zmat_angle_variables(ZMAT, ref_angle_variables):
    assert list(ZMAT.angle_variables) == ref_angle_variables


@pytest.mark.parametrize(
    "ZMAT, ref_tors_indices",
    ref_tors_indices,
    ids=["standard zmat", "automatic delocalized", "custom"],
)
def test_zmat_torsion_indices(ZMAT, ref_tors_indices):
    for i in range(len(ZMAT.torsion_indices)):
        for j in range(len(ZMAT.torsion_indices[i])):
            assert ZMAT.torsion_indices[i][j] == ref_tors_indices[i][j]


@pytest.mark.parametrize(
    "ZMAT, ref_tors_variables",
    ref_tors_variables,
    ids=["standard zmat", "automatic delocalized", "custom"],
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
        "delocalized out of plane indices",
        "delocalized out of place variables",
        "delocalized lin indices",
        "delocalized lin variables",
        "delocalized linx indices",
        "delocalized linx variables",
        "delocalized_liny indices",
        "delocalized_liny variables",
    ],
)
def test_custom_zmat(custom_zmat_coords, reference_coords):
    for i in range(len(custom_zmat_coords)):
        for j in range(len(custom_zmat_coords[i])):
            assert custom_zmat_coords[i][j] == reference_coords[i][j]


# Only need to test the Custom internal coordinates
# TODO pythonize
def test_zmat_calc():
    os.chdir("./ref_data/zmat_test/")
    options = Options()
    errors = []

    options.coords = "Custom"
    ZMAT = Zmat(options)
    output_test_del = ZMAT.zmat_read("zmat_custom")
    ZMAT.zmat_process(output_test_del)

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
    var_dict_custom = ZMAT.variable_dictionary_a
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
    ZMAT.run("zmat_custom")

    index_dict_ref = {
        "R1": np.array(["1", "2"], dtype=object),
        "R2": np.array(["1", "3"], dtype=object),
        "R3": np.array(["1", "4"], dtype=object),
        "R4": np.array(["1", "5"], dtype=object),
        "R5": np.array(["2", "6"], dtype=object),
        "A1": np.array(["2", "1", "3"], dtype=object),
        "A2": np.array(["2", "1", "4"], dtype=object),
        "A3": np.array(["2", "1", "5"], dtype=object),
        "A4": np.array(["3", "1", "4"], dtype=object),
        "A5": np.array(["4", "1", "5"], dtype=object),
        "A6": np.array(["5", "1", "3"], dtype=object),
        "A7": np.array(["6", "2", "1"], dtype=object),
        "D1": np.array(["6", "2", "1", "4"], dtype=object),
        "O1": np.array(["4", "1", "3", "5"], dtype=object),
        "L1": np.array(["5", "1", "2", "4"], dtype=object),
        "Lx1": np.array(["3", "1", "2", "6"], dtype=object),
        "Ly1": np.array(["3", "1", "2", "6"], dtype=object),
    }
    index_dict_custom = ZMAT.index_dictionary

    errors = []
    for key in index_dict_ref:
        for i in range(len(index_dict_ref[key])):
            for j in range(len(index_dict_ref[key][i])):
                assert index_dict_ref[key][i][j] == index_dict_custom[key][i][j]
