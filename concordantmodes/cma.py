import os
import re
import sys
import shutil
import subprocess
import time
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
from concordantmodes.algorithm import Algorithm
from concordantmodes.directory_tree import DirectoryTree
from concordantmodes.f_convert import FcConv
from concordantmodes.f_read import FcRead
from concordantmodes.force_constant import ForceConstant
from concordantmodes.gf_method import GFMethod
from concordantmodes.g_matrix import GMatrix
from concordantmodes.int2cart import Int2Cart
from concordantmodes.reap import Reap
from concordantmodes.rmsd import RMSD
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.zmat import Zmat

# from concordantmodes.second_order import SecOrder


class ConcordantModes(object):
    """
    This constant is from:
    https://physics.nist.gov/cgi-bin/cuu/Value?hr
    If this link dies, find the new link on NIST
    for the Hartree to Joule conversion and pop it in there.
    There is a standard uncertainty of 0.0000000000085 to the MDYNE_HART constant.
    BOHR_ANG: Standard uncertainty of 0.00000000080
    """

    def __init__(self, options, proj=None):
        self.options = options
        self.MDYNE_HART = 4.3597447222071
        self.BOHR_ANG = 0.529177210903
        self.proj = proj

    def run(self):
        t1 = time.time()

        rootdir = os.getcwd()

        # Parse the output to get all pertinent ZMAT info
        self.zmat_obj = Zmat(self.options)
        self.zmat_obj.run()
        if self.options.geom_check:
            raise RuntimeError

        # Compute the initial s-vectors
        s_vec = SVectors(
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_init
        )
        if self.options.man_proj:
            proj = self.proj
            print(proj.shape)
            print(proj)
            s_vec.run(
                self.zmat_obj.cartesians_init,
                True,
                proj=proj,
                second_order=self.options.second_order,
            )
        else:
            s_vec.run(
                self.zmat_obj.cartesians_init,
                True,
                second_order=self.options.second_order,
            )

        self.TED_obj = TED(s_vec.proj, self.zmat_obj)

        # Print out the percentage composition of the projected coordinates
        if self.options.coords != "ZMAT":
            self.TED_obj.run(
                np.eye(len(self.TED_obj.proj.T)), np.zeros(len(self.TED_obj.proj.T))
            )

        # Compute G-Matrix
        g_mat = GMatrix(self.zmat_obj, s_vec, self.options)
        g_mat.run()

        G = g_mat.G.copy()

        # Read in FC matrix in cartesians, then convert to internals.
        # Or compute an initial hessian in internal coordinates.
        init_bool = False
        if os.path.exists(rootdir + "/fc.dat"):
            f_read_obj = FcRead("fc.dat")
        elif os.path.exists(rootdir + "/FCMFINAL"):
            f_read_obj = FcRead("FCMFINAL")
        else:
            init_bool = True

            # First generate displacements in internal coordinates
            eigs_init = np.eye(len(s_vec.proj.T))
            if not self.options.deriv_level_init:
                indices = np.triu_indices(len(s_vec.proj.T))
                indices = np.array(indices).T
            else:
                indices = np.arange(len(eigs_init))

            init_disp = TransfDisp(
                s_vec,
                self.zmat_obj,
                self.options.disp,
                eigs_init,
                True,
                self.options.disp_tol,
                self.TED_obj,
                self.options,
                indices,
                deriv_level=self.options.deriv_level_init,
            )
            init_disp.run()

            prog_init = self.options.program_init
            prog_name_init = prog_init.split("@")[0]

            if self.options.gen_disps_init:
                if os.path.exists(rootdir + "/DispsInit"):
                    if os.path.exists(rootdir + "/oldDispsInit"):
                        shutil.rmtree(rootdir + "/oldDispsInit")
                    shutil.copytree(rootdir + "/DispsInit", rootdir + "/oldDispsInit")
                    shutil.rmtree(rootdir + "/DispsInit")

                dir_obj_init = DirectoryTree(
                    prog_name_init,
                    self.zmat_obj,
                    init_disp,
                    self.options.cart_insert_init,
                    init_disp.p_disp,
                    init_disp.m_disp,
                    self.options,
                    indices,
                    "templateInit.dat",
                    "DispsInit",
                    deriv_level=self.options.deriv_level_init,
                )
                dir_obj_init.run()

                if not self.options.calc_init:
                    print(
                        "The initial displacements have been generated, now they must be run locally."
                    )
                    raise RuntimeError
            else:
                if not os.path.exists(rootdir + "/DispsInit"):
                    print(
                        "You need to have a DispsInit directory already present if you want to proceed under current conditions!"
                    )
                    raise RuntimeError

            os.chdir(rootdir + "/DispsInit")

            if self.options.calc_init:
                disp_list = []
                for i in os.listdir(rootdir + "/DispsInit"):
                    disp_list.append(i)

                if self.options.cluster != "sapelo":
                    v_template = VulcanTemplate(
                        self.options, len(disp_list), prog_name_init, prog_init
                    )
                    out = v_template.run()
                    with open("displacements.sh", "w") as file:
                        file.write(out)

                    # Submits an array, then checks if all jobs have finished every
                    # 10 seconds.
                    sub = Submit(disp_list, self.options)
                    sub.run()
                else:
                    s_template = SapeloTemplate(
                        self.options, len(disp_list), prog_name_init, prog_init
                    )
                    out = s_template.run()
                    with open("optstep.sh", "w") as file:
                        file.write(out)
                    for z in range(0, len(disp_list)):
                        source = os.getcwd() + "/optstep.sh"
                        os.chdir("./" + str(z + 1))
                        destination = os.getcwd()
                        shutil.copy2(source, destination)
                        os.chdir("../")
                    sub = Submit(disp_list, self.options)
                    sub.run()

            reap_obj_init = Reap(
                prog_name_init,
                self.zmat_obj,
                init_disp.disp_cart,
                self.options,
                init_disp.n_coord,
                eigs_init,
                indices,
                self.options.energy_regex_init,
                self.options.gradient_regex,
                self.options.molly_regex_init,
                self.options.success_regex_init,
                deriv_level=self.options.deriv_level_init,
                # disp_sym = init_disp.disp_sym
            )
            reap_obj_init.run()

            if not self.options.deriv_level_init:
                p_array_init = reap_obj_init.p_en_array
                m_array_init = reap_obj_init.m_en_array
                ref_en_init = reap_obj_init.ref_en
                deriv_level = 0
            else:
                cart_p_array_init = reap_obj_init.p_grad_array
                cart_m_array_init = reap_obj_init.m_grad_array
                p_array_init = np.zeros(np.eye(len(eigs_init)).shape)
                m_array_init = np.zeros(np.eye(len(eigs_init)).shape)
                ref_en_init = None

                # Need to convert this array here from cartesians to internals using projected A-tensor
                for i in indices:
                    grad_s_vec = SVectors(
                        self.zmat_obj,
                        self.options,
                        self.zmat_obj.variable_dictionary_init,
                    )
                    grad_s_vec.run(init_disp.p_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.TED_obj.proj)
                    p_array_init[i] = np.dot(cart_p_array_init[i].T, A_proj)
                    grad_s_vec.run(init_disp.m_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.TED_obj.proj)
                    m_array_init[i] = np.dot(cart_m_array_init[i].T, A_proj)

                deriv_level = 1

            fc_init = ForceConstant(
                init_disp,
                p_array_init,
                m_array_init,
                ref_en_init,
                self.options,
                indices,
                deriv_level=deriv_level,
            )
            fc_init.run()
            print("Computed Force Constants:")
            print(fc_init.FC)

        # Temporary code to ensure nothing breaks in my code in the meantime
        self.options.deriv_level_init = 0
        self.options.deriv_level = 0

        if not init_bool:
            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                s_vec,
                self.zmat_obj,
                "internal",
                False,
                self.TED_obj,
                self.options.units,
            )
            f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC

        if self.options.coords != "ZMAT" and not init_bool:
            F = np.dot(self.TED_obj.proj.T, np.dot(F, self.TED_obj.proj))
        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(self.TED_obj.proj.T, np.dot(g_mat.G, self.TED_obj.proj))

        # Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
        print("Initial Frequencies:")
        init_GF = GFMethod(
            g_mat.G.copy(),
            F.copy(),
            self.options.tol,
            self.options.proj_tol,
            self.zmat_obj,
            self.TED_obj,
            cma="init",
        )
        init_GF.run()

        # Now for the TED check.
        self.G = np.dot(np.dot(LA.inv(init_GF.L), g_mat.G), LA.inv(init_GF.L).T)
        self.G[np.abs(self.G) < self.options.tol] = 0
        self.F = np.dot(np.dot(init_GF.L.T, F), init_GF.L)
        self.F[np.abs(self.F) < self.options.tol] = 0

        print("TED Frequencies:")
        TED_GF = GFMethod(
            self.G,
            self.F,
            self.options.tol,
            self.options.proj_tol,
            self.zmat_obj,
            self.TED_obj,
            cma=False,
        )
        TED_GF.run()

        # S = TED_GF.S
        initial_fc = TED_GF.eig_v
        eigs = len(TED_GF.S)

        algo = Algorithm(eigs, initial_fc, self.options)
        algo.run()
        print(algo.indices)

        # Recompute the B-Tensors to match the final geometry,
        # then generate the displacements.

        s_vec = SVectors(
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_final
        )
        s_vec.run(self.zmat_obj.cartesians_final, False, proj=self.TED_obj.proj)

        transf_disp = TransfDisp(
            s_vec,
            self.zmat_obj,
            self.options.disp,
            init_GF.L,
            True,
            self.options.disp_tol,
            self.TED_obj,
            self.options,
            algo.indices,
            GF=TED_GF,
        )
        # B_tensor = s_vec.B
        # second_B = SecOrder(
        #    self.zmat_obj, self.zmat_obj.cartesians_final, B_tensor, self.options
        # )
        # second_B.run()
        transf_disp.run()
        p_disp = transf_disp.p_disp
        m_disp = transf_disp.m_disp
        if self.options.disp_check:
            raise RuntimeError

        # The displacements have been generated, now we have to run them!

        prog = self.options.program
        progname = prog.split("@")[0]
        if self.options.gen_disps:
            if os.path.exists(rootdir + "Disps"):
                if os.path.exists(rootdir + "/oldDisps"):
                    shutil.rmtree(rootdir + "/oldDisps")
                shutil.copytree(rootdir + "/Disps", rootdir + "/oldDisps")
                shutil.rmtree(rootdir + "/Disps")
            dir_obj = DirectoryTree(
                progname,
                self.zmat_obj,
                transf_disp,
                self.options.cart_insert,
                p_disp,
                m_disp,
                self.options,
                algo.indices,
                "template.dat",
                "Disps",
            )
            dir_obj.run()

            if not self.options.calc:
                print(
                    "The displacements have been generated, now they must be run locally."
                )
                raise RuntimeError
        else:
            if not os.path.exists(rootdir + "/Disps"):
                print(
                    "You need to have a Disps directory already present if you want to proceed under current conditions!"
                )
                raise RuntimeError

        os.chdir(rootdir + "/Disps")

        if self.options.calc:
            disp_list = []
            for i in os.listdir(rootdir + "/Disps"):
                disp_list.append(i)
            print("printing disp list")
            print(disp_list)

            # Generates the submit script for the displacements.
            if self.options.cluster != "sapelo":
                v_template = VulcanTemplate(
                    self.options, len(disp_list), progname, prog
                )
                out = v_template.run()
                with open("displacements.sh", "w") as file:
                    file.write(out)

                # Submits an array, then checks if all jobs have finished every
                # 10 seconds.
                sub = Submit(disp_list, self.options)
                sub.run()
            else:
                s_template = SapeloTemplate(
                    self.options, len(disp_list), progname, prog
                )
                out = s_template.run()
                with open("optstep.sh", "w") as file:
                    file.write(out)
                for z in range(0, len(disp_list)):
                    source = os.getcwd() + "/optstep.sh"
                    os.chdir("./" + str(z + 1))
                    destination = os.getcwd()
                    shutil.copy2(source, destination)
                    os.chdir("../")
                sub = Submit(disp_list, self.options)
                sub.run()

        # After this point, all of the jobs will have finished, and its time
        # to reap the energies as well as checking for sucesses on all of
        # the jobs
        print(eigs)
        reap_obj = Reap(
            progname,
            self.zmat_obj,
            transf_disp.disp_cart,
            self.options,
            transf_disp.n_coord,
            eigs,
            algo.indices,
            self.options.energy_regex,
            self.options.gradient_regex,
            self.options.molly_regex_init,
            self.options.success_regex,
            # disp_sym = transf_disp.disp_sym
        )
        reap_obj.run()

        p_en_array = reap_obj.p_en_array
        m_en_array = reap_obj.m_en_array
        ref_en = reap_obj.ref_en

        fc = ForceConstant(
            transf_disp, p_en_array, m_en_array, ref_en, self.options, algo.indices
        )
        fc.run()
        print("Computed Force Constants:")
        print(fc.FC)

        self.F = fc.FC
        if self.options.benchmark_full:
            with open("Full_fc_levelB.npy", "wb") as z:
                np.save(z, fc.FC)

        # Recompute the G-matrix with the new geometry, and then transform
        # the G-matrix using the lower level of theory eigenvalue matrix.
        # This will not fully diagonalize the G-matrix if a different
        # geometry is used between the two.

        g_mat = GMatrix(self.zmat_obj, s_vec, self.options)
        g_mat.run()

        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(self.TED_obj.proj.T, np.dot(g_mat.G, self.TED_obj.proj))

        self.G = np.dot(np.dot(transf_disp.eig_inv, g_mat.G), transf_disp.eig_inv.T)
        self.G[np.abs(self.G) < self.options.tol] = 0

        if self.options.benchmark_full:
            cma = True
        else:
            cma = False

        # Final GF Matrix run
        print("Final Frequencies:")
        final_GF = GFMethod(
            self.G,
            self.F,
            self.options.tol,
            self.options.proj_tol,
            self.zmat_obj,
            self.TED_obj,
            cma=cma,
        )
        final_GF.run()

        # This code below is a rudimentary table of the TED for the final
        # frequencies. Actually right now it uses the initial L-matrix,
        # which may need to be modified by the final L-matrix.
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Final TED"))
        print("////////////////////////////////////////////")
        self.TED_obj.run(np.dot(init_GF.L, final_GF.L), final_GF.freq)

        # This code prints out the frequencies in order of energy as well
        # as the ZPVE in several different units.
        print(
            "Final ZPVE in: "
            + "{:6.2f}".format(np.sum(final_GF.freq) / 2)
            + " (cm^-1) "
            + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 349.7550881133)
            + " (kcal mol^-1) "
            + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 219474.6313708)
            + " (hartrees) "
        )

        # This code converts the force constants back into cartesian
        # coordinates and writes out an "output.default.hess" file, which
        # is of the same format as FCMFINAL of CFOUR.
        self.F = np.dot(np.dot(transf_disp.eig_inv.T, self.F), transf_disp.eig_inv)
        if self.options.coords != "ZMAT":
            self.F = np.dot(self.TED_obj.proj, np.dot(self.F, self.TED_obj.proj.T))
        cart_conv = FcConv(
            self.F,
            s_vec,
            self.zmat_obj,
            "cartesian",
            True,
            self.TED_obj,
            self.options.units,
        )
        cart_conv.run()

        # if mol2.size != 0:
        #    print('I cant believe its not CMA1!')
        #    if self.options.rmsd:
        #
        #        rmsd_geom = RMSD()
        #        rmsd_geom.run(mol1,mol2)

        t2 = time.time()
        print("Frequency Shift (cm^-1): ")
        print(final_GF.freq - init_GF.freq)
        print("This program took " + str(t2 - t1) + " seconds to run.")

        # after this point, you could loop through the full FC matrix and compute several other benchmark frequencies
        if self.options.benchmark_full:
            self.options.off_diag = True
            # nate
            with open("Full_fc_levelB.npy", "rb") as z:
                FC = np.load(z)
            self.F = np.zeros((eigs, eigs))
            for q in range(eigs):
                self.F = np.zeros((eigs, eigs))
                self.options.off_diag_bands = q
                a = eigs
                algo = Algorithm(eigs, initial_fc, self.options)
                algo.run()
                print(algo.indices)
                if q == 0:
                    cma = None
                else:
                    cma = False
                for index in algo.indices:
                    i, j = index[0], index[1]
                    self.F[i, j] = FC[i, j]
                cf = np.triu_indices(a, 1)
                il = (cf[1], cf[0])
                self.F[il] = self.F[cf]
                # Final GF Matrix run
                print("Final Frequencies:")
                final_GF = GFMethod(
                    self.G,
                    self.F,
                    self.options.tol,
                    self.options.proj_tol,
                    self.zmat_obj,
                    self.TED_obj,
                    cma=cma,
                )
                final_GF.run()

                # This code below is a rudimentary table of the TED for the final
                # frequencies. Actually right now it uses the initial L-matrix,
                # which may need to be modified by the final L-matrix.
                print("////////////////////////////////////////////")
                print("//{:^40s}//".format(" Final TED"))
                print("////////////////////////////////////////////")
                self.TED_obj.run(np.dot(init_GF.L, final_GF.L), final_GF.freq)

                # This code prints out the frequencies in order of energy as well
                # as the ZPVE in several different units.
                print(
                    "Final ZPVE in: "
                    + "{:6.2f}".format(np.sum(final_GF.freq) / 2)
                    + " (cm^-1) "
                    + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 349.7550881133)
                    + " (kcal mol^-1) "
                    + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 219474.6313708)
                    + " (hartrees) "
                )

                # This code converts the force constants back into cartesian
                # coordinates and writes out an "output.default.hess" file, which
                # is of the same format as FCMFINAL of CFOUR.
                # self.F = np.dot(np.dot(transf_disp.eig_inv.T, self.F), transf_disp.eig_inv)
                # if self.options.coords != "ZMAT":
                #    self.F = np.dot(self.TED_obj.proj, np.dot(self.F, self.TED_obj.proj.T))
                # cart_conv = FcConv(
                #    self.F,
                #    s_vec,
                #    self.zmat_obj,
                #    "cartesian",
                #    True,
                #    self.TED_obj,
                #    self.options.units,
                # )
                # cart_conv.run()

                t2 = time.time()
                print("Frequency Shift (cm^-1): ")
                print(final_GF.freq - init_GF.freq)
                print("This program took " + str(t2 - t1) + " seconds to run.")

                print("////////////////////////////////////////////")
                print("//{:^40s}//".format(" END OF LOOP"))
                print("////////////////////////////////////////////")

            print("////////////////////////////////////////////")
            print("//{:^40s}//".format("Begin Selective Diagnostic Benchmark"))
            print("////////////////////////////////////////////")
            self.options.off_diag = False
            self.options.mode_coupling_check = True

            algo = Algorithm(eigs, initial_fc, self.options)
            algo.run()
            print("printing algo indices", algo.indices)
            diagnostic_indices = algo.indices
            # with open("L_full.npy", "rb") as w:
            # L_full = np.load(w)
            # with open("L_0.npy", "rb") as x:
            # L_0 = np.load(x)

            # L_inv = LA.inv(L_full)
            # S = np.dot(L_inv, L_0)
            # print("printing overlap")
            # print(S)
            # with open("S.npy", "wb") as y:
            #    np.save(y, S)
            self.options.mode_coupling_check = False
            algo = Algorithm(eigs, initial_fc, self.options)
            algo.run()
            newlist = algo.indices + diagnostic_indices
            print(newlist)

            for index in newlist:
                i, j = index[0], index[1]
                self.F[i, j] = FC[i, j]
            cf = np.triu_indices(a, 1)
            il = (cf[1], cf[0])
            self.F[il] = self.F[cf]
            # Final GF Matrix run
            print("Final Frequencies:")
            cma = False
            final_GF = GFMethod(
                self.G,
                self.F,
                self.options.tol,
                self.options.proj_tol,
                self.zmat_obj,
                self.TED_obj,
                cma=cma,
            )
            final_GF.run()

            # This code below is a rudimentary table of the TED for the final
            # frequencies. Actually right now it uses the initial L-matrix,
            # which may need to be modified by the final L-matrix.
            print("////////////////////////////////////////////")
            print("//{:^40s}//".format(" Final TED"))
            print("////////////////////////////////////////////")
            self.TED_obj.run(np.dot(init_GF.L, final_GF.L), final_GF.freq)
            # if self.options.clean_house:
            # os.system("rm L_full.npy")

            # This code prints out the frequencies in order of energy as well
            # as the ZPVE in several different units.
            print(
                "Final ZPVE in: "
                + "{:6.2f}".format(np.sum(final_GF.freq) / 2)
                + " (cm^-1) "
                + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 349.7550881133)
                + " (kcal mol^-1) "
                + "{:6.2f}".format(0.5 * np.sum(final_GF.freq) / 219474.6313708)
                + " (hartrees) "
            )
            # if self.options.clean_house:
            # os.system("rm L_0.npy L_full.npy S.npy Full_fc_levelB.npy")
        # if self.options.clean_house:
        # os.system("rm S_p.npy")
        # path = os.getcwd() + "/L_full.npy"
        # if os.path.isfile(path):
        # os.system("rm L_full.npy")
