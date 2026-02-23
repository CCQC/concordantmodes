import fileinput
import math
import os
import re
import shutil
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA

from .suite_execute import execute_suite

from concordantmodes.algorithm import Algorithm
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.options import Options
from concordantmodes.reap import Reap
from concordantmodes.s_vectors import SVectors
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.zmat import Zmat

suite = execute_suite("./ref_data/reap_test/", "Delocalized")
suite.run()


def test_h2o():
    # In principle I can test everything integratively in this module outside of the submit function.
    # It doesn't really make sense to me that github would be able to run the displacements to verify the singlepoints.
    # Perhaps I start from just carts, generate auto-deloc coords, follow the process through until displacements are generated.
    # Then I check if the displacements match up with the reference, move on to pull out single point energies, then compare
    # frequencies at the end. I can actually do this in a two step process at the computation of Level B and Level A.

    suite.options.program_a = "psi4@master"
    prog = suite.options.program_a
    prog_name = prog.split("@")[0]

    suite.options.energy_regex_a = r"Giraffe The Energy is\s+(\-\d+\.\d+)"
    suite.options.success_regex_a = r"beer"
    os.chdir(suite.path + "/Disps")
    suite.algo.indices = [[0, 0], [1, 1], [2, 2]]
    reap_obj = Reap(
        suite.options,
        len(suite.GF.L),
        suite.algo.indices,
        suite.symm_obj,
        "A",
        deriv_level=suite.options.deriv_level_a,
    )
    reap_obj.run()

    # ref_en = -76.332189646734

    os.chdir("../..")
    os.chdir(suite.root)
    assert math.isclose(, , rel_tol=0.0, abs_tol=1e-10)
    # assert math.isclose(ref_en, reap_obj.m_en_array[1][1], rel_tol=0.0, abs_tol=1e-10)
