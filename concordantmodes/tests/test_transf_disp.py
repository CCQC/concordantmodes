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
        [-1.375074, -0.024277, -0.001321],
        [ 1.306015,  0.120701,  0.000598],
        [-2.077867,  1.909611,  0.002537],
        [-2.102861, -0.978391,  1.683774],
        [-2.115089, -0.980281, -1.680084],
        [ 1.941174, -1.577492,  0.000003],
    ]

    os.chdir(suite.root)

    print(np.asarray(disp_ref))
    print(disps.m_disp[3][3])

    assert np.allclose(np.asarray(disp_ref), disps.m_disp[3][3], rtol=0.0, atol=1e-5)


test_transf_disp()
