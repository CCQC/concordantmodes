import os
import shutil
import numpy as np
from concordantmodes.ted import TED
from numpy.linalg import inv
from numpy import linalg as LA

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


class execute_suite(object):
    def __init__(self, path, coords, s_vec_bool=False, disp_transf=False):
        self.path = path
        self.coords = coords
        self.s_vec_bool = s_vec_bool
        self.disp_transf = disp_transf

    def run(self):
        os.chdir(self.path)
        self.options = Options()
        # self.options.coords = "Redundant"
        self.options.coords = self.coords
        self.ZMAT = Zmat(self.options)
        output_test = self.ZMAT.zmat_read("zmat")
        self.ZMAT.zmat_process(output_test)

        self.ZMAT.zmat_calc()

        self.ZMAT.zmat_compile()

        self.s_vec = SVectors(
            self.ZMAT, self.options, self.ZMAT.variable_dictionary_init
        )
        self.s_vec.run(self.ZMAT.cartesians_init, True)

        self.TED_obj = TED(self.s_vec.proj, self.ZMAT)
        self.g_mat = GMatrix(self.ZMAT, self.s_vec, self.options)
        self.g_mat.run()
        if self.s_vec_bool:
            print("It ran")
            os.chdir("../../")
            return
        self.FC = FcRead("fc.dat")
        self.FC.run()
        self.f_conv = FcConv(
            self.FC.fc_mat,
            self.s_vec,
            self.ZMAT,
            "internal",
            False,
            self.TED_obj,
            self.options.units,
        )
        self.f_conv.run()

        self.F = np.dot(self.TED_obj.proj.T, np.dot(self.f_conv.F, self.TED_obj.proj))
        self.G = np.dot(self.TED_obj.proj.T, np.dot(self.g_mat.G, self.TED_obj.proj))
        self.GF = GFMethod(
            self.G,
            self.F,
            self.options.tol,
            self.options.proj_tol,
            self.ZMAT,
            self.TED_obj,
        )

        self.GF.run()
        self.algo = Algorithm(len(self.GF.L), None, self.options)
        self.algo.run()

        self.disps = TransfDisp(
            self.s_vec,
            self.ZMAT,
            self.options.disp,
            self.GF.L,
            True,
            self.options.disp_tol,
            self.TED_obj,
            self.options,
            self.algo.indices,
        )
        self.disps.run()
        os.chdir("../../")
