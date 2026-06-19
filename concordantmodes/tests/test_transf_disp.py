import os
import shutil
import numpy as np
from concordantmodes.ted import TED
from numpy.linalg import inv
from numpy import linalg as LA

from .suite_execute import execute_suite

from concordantmodes.algorithm import Algorithm
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.options import Options
from concordantmodes.s_vectors import SVectors
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.zmat import Zmat


def test_transf_disp():
    suite = execute_suite("./ref_data/f_read_test/", "Delocalized")
    suite.run()

    errors = []
    coord_type = "internal"

    disps = TransfDisp(
        suite.s_vec,
        suite.ZMAT,
        suite.GF.L,
        True,
        suite.TED_obj.proj,
        suite.options,
        suite.algo.indices,
        symm_obj=suite.symm_obj,
    )
    disps.run()

    disp_ref = [
        [-1.375077, -0.024277, -0.005898],
        [1.305992, 0.120699, 0.004699],
        [-2.077869, 1.909601, 0.001539],
        [-2.100907, -0.978713, 1.679819],
        [-2.117020, -0.979951, -1.684038],
        [1.941157, -1.577492, 0.003869],
    ]

    os.chdir(suite.root)

    assert np.allclose(np.asarray(disp_ref), disps.m_disp[3][3], rtol=0.0, atol=1e-5)


test_transf_disp()
