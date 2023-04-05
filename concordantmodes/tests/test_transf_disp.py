import os
import shutil
import numpy as np
from concordantmodes.ted import TED
from numpy.linalg import inv
from numpy import linalg as LA

from suite_execute import execute_suite

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
    print(os.getcwd())
    suite = execute_suite("./ref_data/f_read_test/", "Redundant")
    suite.run()

    errors = []

    disps = TransfDisp(
        suite.s_vec,
        suite.ZMAT,
        suite.options.disp,
        suite.GF.L,
        True,
        suite.options.disp_tol,
        suite.TED_obj,
        suite.options,
        suite.algo.indices,
    )
    disps.run()

    disp_ref = [
        [-1.3750734515947842, -0.024279474843673623, 0.0029358198799196837],
        [1.3060102262375441, 0.1207018145868287, -0.0023356196755552655],
        [-2.0778682947484093, 1.9096086118030944, -0.0008923821222545903],
        [-2.1129906861119383, -0.9795514202073832, 1.683039582462003],
        [-2.1049748848658, -0.979121064823173, -1.6808319627150445],
        [1.941172931083388, -1.5774905865156932, -0.0019238478290682895],
    ]

    assert np.allclose(np.asarray(disp_ref), disps.p_disp[3][3], rtol=0.0, atol=1e-8)
