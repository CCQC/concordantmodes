import numpy as np
import os
import re
from numpy.linalg import inv
from numpy import linalg as LA

from .suite_execute import execute_suite

from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.options import Options
from concordantmodes.s_vectors import SVectors
from concordantmodes.ted import TED
from concordantmodes.zmat import Zmat

np.set_printoptions(precision=9)

suite = execute_suite("./ref_data/f_conv_test/", "Delocalized")
suite.run()


def test_f_convert2int():
    errors = []

    FCint = FcConv(
        suite.FC.fc_mat,
        suite.s_vec,
        suite.ZMAT,
        "internal",
        False,
        suite.TED_obj.proj,
        suite.options,
    )
    FCint.run()
    FCint.F = np.dot(suite.TED_obj.proj, np.dot(FCint.F, suite.TED_obj.proj.T))

    FCintR = FcRead(suite.path + "/fc_int.dat")
    FCintR.run()
    os.chdir(suite.root)
    assert np.allclose(FCint.F, FCintR.fc_mat, rtol=0, atol=1e-10)


def test_f_convert2cart():
    errors = []

    FCint = FcConv(
        suite.FC.fc_mat,
        suite.s_vec,
        suite.ZMAT,
        "internal",
        False,
        suite.TED_obj.proj,
        suite.options,
    )
    FCint.run()
    FCcart = FcConv(
        FCint.F,
        suite.s_vec,
        suite.ZMAT,
        "cartesian",
        False,
        suite.TED_obj.proj,
        suite.options,
    )
    FCcart.run()

    FCintC = FcRead(suite.path + "/fc_cart.dat")
    FCintC.run()

    os.chdir(suite.root)
    assert np.allclose(FCcart.F, FCintC.fc_mat, rtol=0.0, atol=1e-10)
