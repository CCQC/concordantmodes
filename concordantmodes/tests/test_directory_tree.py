import fileinput
import os
import re
import shutil
import numpy as np

from concordantmodes.directory_tree import DirectoryTree
from concordantmodes.options import Options
from concordantmodes.symmetry import Symmetry
from concordantmodes.zmat import Zmat


def test_make_input():
    
    os.chdir("./ref_data/dir_tree/")

    options = Options()
    options.cart_insert_a = 9

    zmat = Zmat(options)
    zmat.zmat_read("zmat")

    symm_obj = Symmetry(zmat, options, np.array([]))
    symm_obj.dummy_obj()
    symm_obj.symtext = None

    dispp = zmat.cartesians_a
    at = zmat.atom_list
    index = options.cart_insert_a

    DT = DirectoryTree(
        "molpro",
        zmat,
        None,
        "A",
        None,
        None,
        options,
        None,
        symm_obj,
        "template.dat",
        None,
    )

    with open("template.dat", "r") as file:
        data = file.readlines()
    with open("template_ref.dat", "r") as file:
        reference = file.readlines()

    if os.path.exists(os.getcwd() + "/1"):
        shutil.rmtree(os.getcwd() + "/1")

    data = DT.make_input(data, dispp, len(at), at, index, "input.dat", "1")
    os.chdir("../..")

    assert data == reference
