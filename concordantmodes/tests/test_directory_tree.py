import fileinput
import os
import re
import shutil
import numpy as np

from concordantmodes.options import Options
from concordantmodes.zmat import Zmat
from concordantmodes.directory_tree import DirectoryTree


def test_make_input():
    os.chdir("./ref_data/dir_tree/")

    options = Options()
    options.cart_insert = 9

    zmat = Zmat(options)
    zmat.zmat_read("zmat")

    dispp = zmat.cartesians_final
    at = zmat.atom_list
    index = options.cart_insert

    DT = DirectoryTree(
        "molpro", zmat, None, None, None, None, options, None, "template.dat", None
    )

    with open("template.dat", "r") as file:
        data = file.readlines()
    with open("template_ref.dat", "r") as file:
        reference = file.readlines()

    data = DT.make_input(data, dispp, len(at), at, index)
    os.chdir("../..")

    assert data == reference
