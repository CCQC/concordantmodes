import os
import re
import sys
import shutil
import subprocess
import time
import numpy as np
import copy
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
from concordantmodes.g_read import GrRead
from concordantmodes.int2cart import Int2Cart
from concordantmodes.molden_writer import MoldenWriter
from concordantmodes.reap import Reap
from concordantmodes.rmsd import RMSD
from concordantmodes.s_vectors import SVectors
from concordantmodes.submit import Submit
from concordantmodes.symmetry import Symmetry
from concordantmodes.ted import TED
from concordantmodes.transf_disp import TransfDisp
from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.zmat import Zmat


class ConcordantModes(object):
    """
    This constant is from:
    https://physics.nist.gov/cgi-bin/cuu/Value?hr
    If this link dies, find the new link on NIST
    for the Hartree to Joule conversion and pop it in there.
    There is a standard uncertainty of 0.0000000000085 to the MDYNE_HART constant.
    BOHR_ANG: Standard uncertainty of 0.00000000080
    """

    def __init__(self, options, proj=None, extra_indices=np.array([])):
        self.options = options
        self.MDYNE_HART = 4.3597447222071
        self.BOHR_ANG = 0.529177210903
        self.proj = proj
        self.extra_indices = extra_indices

    def run(self, sym_sort = []):
        t1 = time.time()

        rootdir = os.getcwd()

        #string value to access "state" of CMA program. Are we in level A, B, C, etc...
        cma_level = "B"

        # Parse the output to get all pertinent ZMAT info
        self.zmat_obj = Zmat(self.options)
        self.zmat_obj.run()
        if self.options.geom_check:
            raise RuntimeError
        
        #Do we want to use molsym_symmetry or "manual" symmetry via sym_sort?
        self.symm_obj = Symmetry(self.zmat_obj, self.options, self.proj)
        if self.options.molsym_symmetry:
            self.symm_obj.run()
        else:
            """
            We won't run the symmetry code, but we'll create a dummy object to be passed as an argument.
            #TODO: This is a hacky way to do this, but it's a quick fix for now. Maybe reincorporate symmetry as a s_vector obj?
            """
            self.symm_obj.dummy_obj()
            self.symm_obj.symtext = None
            #check if sym_sort object was passed in. If so, intialize sym_sort objects 
            if len(sym_sort) > 1:
                self.symm_obj.create_flat_sym_sort(sym_sort)

        # Compute the initial s-vectors
        s_vec = SVectors(
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_init
        )
        if self.options.man_proj:
            proj = self.proj
            np.set_printoptions(precision=4, linewidth=240)
            # print(proj.shape)
            # print(proj)
            s_vec.run(
                self.zmat_obj.cartesians_init,
                True,
                proj=proj,
                second_order=self.options.second_order,
            )
            s_vec.proj = proj
        else:
            s_vec.run(
                self.zmat_obj.cartesians_init,
                True,
                second_order=self.options.second_order,
            )
        
        if self.options.molsym_symmetry:
            self.symm_obj.make_proj(s_vec)
            s_vec.proj = copy.deepcopy(self.symm_obj.salc_proj)
        
        self.TED_obj = TED(s_vec.proj, self.zmat_obj, self.options)

        # Print out the percentage composition of the projected coordinates
        if self.options.coords != "ZMAT":
            self.TED_obj.run(
                np.eye(len(self.TED_obj.proj.T)), np.zeros(len(self.TED_obj.proj.T))
            )

        # Compute G-Matrix
        g_mat = GMatrix(self.zmat_obj, s_vec, self.options)
        g_mat.run()

        G = g_mat.G.copy()

        if os.path.exists(rootdir + "/fc.grad"):
            g_read_obj = GrRead("fc.grad")
            g_read_obj.run(self.zmat_obj.cartesians_init)
            print(self.zmat_obj.cartesians_init)

        num_deg_free = s_vec.proj.shape[1]
        # Read in FC matrix in cartesians, then convert to internals.
        # Or compute an initial hessian in internal coordinates.
        self.options.init_bool = False
        if os.path.exists(rootdir + "/fc.dat"):
            f_read_obj = FcRead("fc.dat")
        elif os.path.exists(rootdir + "/FCMFINAL"):
            f_read_obj = FcRead("FCMFINAL")
        else:
            self.options.init_bool = True
            

            # First generate displacements in internal coordinates
            eigs_init = np.eye(len(s_vec.proj.T))
            coord_type = "internal"
            if self.options.cart_fc_init:
                eigs_init = np.eye(len(self.zmat_obj.cartesians_init.flatten()))
                coord_type = "cartesian"
            if not self.options.deriv_level_init:
                if not self.options.molsym_symmetry:
                    algo = Algorithm(num_deg_free, cma_level, self.options)
                    algo.run()
                    # Sym_sort doesn't seem to be working
                    print("symmetric displacements:")
                    if len(sym_sort) > 1:
                        algo.indices = self.symm_obj.create_sym_sort_disps(sym_sort, algo.indices)
                        # indices = self.symm_obj.create_sym_sort_disps(sym_sort, algo.indices)
                else:
                    algo = Algorithm(num_deg_free, cma_level, self.options, proj_irreps=self.symm_obj.proj_irreps)
                    algo.run()
                    self.symm_obj.indices_by_irrep = algo.indices_by_irrep
            else:
                self.symm_obj.proj_irreps = None
                algo = Algorithm(num_deg_free, cma_level, self.options, proj_irreps=self.symm_obj.proj_irreps)
                algo.run()
                self.symm_obj.indices_by_irrep = algo.indices_by_irrep
            
            init_disp = TransfDisp(
                s_vec,
                self.zmat_obj,
                eigs_init,
                True,
                self.TED_obj,
                self.options,
                algo.indices,
                symm_obj=self.symm_obj,
                coord_type=coord_type,
                deriv_level=self.options.deriv_level_init,
                cma_level=cma_level
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
                
                ref_geom_init = init_disp.disp_cart["ref"]

                dir_obj_init = DirectoryTree(
                    prog_name_init,
                    self.zmat_obj,
                    ref_geom_init,
                    cma_level,
                    init_disp.p_disp,
                    init_disp.m_disp,
                    self.options,
                    algo.indices,
                    self.symm_obj,
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
                    # 30 seconds.
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
                self.options,
                len(eigs_init),
                # eigs_init,
                algo.indices,
                self.symm_obj,
                cma_level,
                deriv_level=self.options.deriv_level_init,
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
                for i in range(len(algo.indices)):
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
                algo.indices,
                deriv_level=deriv_level,
                coord_type_init=coord_type,
                cma_level=cma_level
            )
            fc_init.run()
            print("Computed Force Constants:")
            print(fc_init.FC)
            if self.options.second_order:
                f_conv_obj = FcConv(
                    fc_init.FC,
                    s_vec,
                    self.zmat_obj,
                    "internal",
                    False,
                    self.TED_obj,
                    self.options,
                )
                f_conv_obj.run(grad=fc_init.gradient)
                fc_init.FC = f_conv_obj.F



        # Temporary code to ensure nothing breaks
        self.options.deriv_level_init = 0
        self.options.deriv_level = 0

        if not self.options.init_bool:
            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                s_vec,
                self.zmat_obj,
                "internal",
                False,
                self.TED_obj,
                self.options,
            )
            if self.options.second_order:
                f_conv_obj.run(grad=g_read_obj.cart_grad)
            else:
                f_conv_obj.run()
            F = f_conv_obj.F
        else:
            F = fc_init.FC

        if self.options.coords != "ZMAT" and not self.options.init_bool:
            F = np.dot(self.TED_obj.proj.T, np.dot(F, self.TED_obj.proj))
        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(self.TED_obj.proj.T, np.dot(g_mat.G, self.TED_obj.proj))

        self.options.init_bool = False
        # print("F and then G:")
        F[np.abs(F) < self.options.tol] = 0
        g_mat.G[np.abs(g_mat.G) < self.options.tol] = 0
        # print(F)
        # print(g_mat.G / (5.48579909065 * (10 ** (-4))))
        
        if len(sym_sort) > 1:
            F, g_mat.G = self.symm_obj.GF_sym_sort(F, g_mat, sym_sort)

        # Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
        print("Initial Frequencies:")
        init_GF = GFMethod(
            g_mat.G.copy(),
            F.copy(),
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma="init",
        )
        init_GF.run()

        # For a quick test, to be deleted later
        g_init = g_mat.G.copy()
        f_init = F.copy()

        ted_b = init_GF.ted.TED
        #more sym_sort stuff
        if len(sym_sort):
            self.irreps_init,flat_sym_freqs = self.symm_obj.mode_symmetry_sort(init_GF.ted.TED,sym_sort,init_GF.freq)
            self.ref_init = np.array(flat_sym_freqs)
            #### this block could probably be moved inside the symmetry.py module? 
            flat_sym_modes_b = [
                x
                for xs in self.irreps_init
                for x in xs
            ]
            print(flat_sym_modes_b)
            del_list = []
            # for i in range(len(self.irreps_init)):
                # if len(self.irreps_init[i]) == 1:
                    # del_list.append(self.irreps_init[i][0])
            flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b),del_list)
            ted_b = ted_b.T
            ted_b = ted_b[flat_sym_modes_b]
            ted_b = ted_b.T
            #### end of block that could probably be moved inside the symmetry.py module? 


        # Now for the TED check.
        self.G = np.dot(np.dot(LA.inv(init_GF.L), g_mat.G), LA.inv(init_GF.L).T)
        self.G[np.abs(self.G) < self.options.tol] = 0
        self.F = np.dot(np.dot(init_GF.L.T, F), init_GF.L)
        self.F[np.abs(self.F) < self.options.tol] = 0

        print("TED Frequencies: Degeneracy x Irrep")
        TED_GF = GFMethod(
            self.G,
            self.F,
            #self.options.tol,
            #self.options.proj_tol,
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma=False,
        )
        TED_GF.run()

        initial_fc = TED_GF.eig_v
        #eigs = len(TED_GF.S)
        #TODO this needs to be defined with respect to the s_vector number degrees of freedom
        #eigs = len(TED_GF.eig_v)
        #print(f"The eigs {eigs}")
        #print(f"The shape of s_vec {s_vec.proj.shape}")
        #print(stop)

        #Now switch state to cma_level = "A"
        cma_level = "A"
        if self.options.molsym_symmetry:
            algo = Algorithm(num_deg_free, cma_level, self.options, self.symm_obj.proj_irreps)
        else:
            algo = Algorithm(num_deg_free, cma_level, self.options, None)
        #algo = Algorithm(eigs, initial_fc, self.options)
        # algo.options.off_diag_bands = 2
        # algo.options.off_diag_limit = False
        # algo.options.off_diag = True

        algo.run()
        if self.options.molsym_symmetry:
            self.symm_obj.indices_by_irrep = algo.indices_by_irrep
        if len(self.extra_indices):
            algo.indices = np.append(algo.indices, self.extra_indices, axis=0)

        # Recompute the B-Tensors to match the final geometry,
        # then generate the displacements.

        s_vec = SVectors(
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_final
        )
        s_vec.run(self.zmat_obj.cartesians_final, False, proj=self.TED_obj.proj)

        transf_disp = TransfDisp(
            s_vec,
            # s_vec.B,
            self.zmat_obj,
            init_GF.L,
            True,
            self.TED_obj,
            self.options,
            algo.indices,
            symm_obj=self.symm_obj,
            cma_level=cma_level
        )
        transf_disp.run(fc=self.F)
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
            
            ref_geom = transf_disp.disp_cart["ref"]
            dir_obj = DirectoryTree(
                progname,
                self.zmat_obj,
                ref_geom,
                cma_level,
                p_disp,
                m_disp,
                self.options,
                algo.indices,
                self.symm_obj,
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

        # After this point, all of the jobs have finished, and it's time
        # to reap the energies as well as checking for sucesses
        reap_obj = Reap(
            self.options,
            num_deg_free,
            algo.indices,
            self.symm_obj,
            cma_level,
        )
        reap_obj.run()

        # raise RuntimeError

        p_en_array = reap_obj.p_en_array
        m_en_array = reap_obj.m_en_array
        ref_en = reap_obj.ref_en


        fc = ForceConstant(
            transf_disp, p_en_array, m_en_array, ref_en, self.options, algo.indices, cma_level=cma_level
        )
        fc.run()
        
        
        np.set_printoptions(precision=4, linewidth=240)
        print("Computed Force Constants:")
        print(fc.FC)
        
        self.F = fc.FC
        f_init = np.dot(np.dot(LA.inv(transf_disp.eig_inv).T, f_init), LA.inv(transf_disp.eig_inv))
        print("Initial FC in same basis:")
        print(f_init)
        # print() 
        # print(ref_en_init)
        # print(ref_en)
        # raise RuntimeError

        # self.F = np.dot(np.dot(LA.inv(init_GF.L).T, F), LA.inv(init_GF.L))
        # self.F = np.dot(np.dot(transf_disp.eig_inv.T, self.F), transf_disp.eig_inv)
        # self.F[np.abs(self.F) < self.options.tol] = 0
        # print("Proj Force Constants:")
        # print(self.F)
        # self.F = np.dot(np.dot(init_GF.L.T, self.F), init_GF.L)
        # self.F[np.abs(self.F) < self.options.tol] = 0
        # print("Normal Mode Force Constants:")
        # print(self.F)

        # Recompute the G-matrix with the new geometry, and then transform
        # the G-matrix using the lower level of theory eigenvalue matrix.
        # This will not fully diagonalize the G-matrix if a different
        # geometry is used between the two.

        g_mat = GMatrix(self.zmat_obj, s_vec, self.options)
        g_mat.run()

        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(self.TED_obj.proj.T, np.dot(g_mat.G, self.TED_obj.proj))

        self.G = g_mat.G
        # print("Proj G:")
        # print(self.G)
        
        self.G = np.dot(np.dot(transf_disp.eig_inv, self.G), transf_disp.eig_inv.T)
        self.G[np.abs(self.G) < self.options.tol] = 0

        # self.G = np.dot(np.dot(LA.inv(init_GF.L), g_mat.G), LA.inv(init_GF.L).T)
        # self.G[np.abs(self.G) < self.options.tol] = 0
        # np.set_printoptions(precision=6, linewidth=240)
        # print("Normal Mode G init:")
        # g_init = np.dot(np.dot(transf_disp.eig_inv, g_init), transf_disp.eig_inv.T)
        # print(g_init)
        # print("Normal Mode G:")
        # print(self.G)
        # raise RuntimeError
        
        # self.G = np.dot(np.dot(LA.inv(transf_disp.eig_inv), g_mat.G), LA.inv(transf_disp.eig_inv).T)
        # print("Normalized G:")
        # print(self.G)
        # print(g_mat.G)
        # self.G = np.dot(np.dot(LA.inv(transf_disp.eig_inv), self.G), LA.inv(transf_disp.eig_inv).T)

        if self.options.benchmark_full:
            cma = True
        else:
            cma = False

        # Final GF Matrix run
        print("Final Harmonic Frequencies:")
        final_GF = GFMethod(
            self.G,
            self.F,
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma=cma,
        )
        final_GF.run()

        # Initial frequency test in normalized eig basis:
        # g_init = np.dot(np.dot(transf_disp.eig_inv, g_init), transf_disp.eig_inv.T)
        

        # print(transf_disp.eig_inv)
        # for i in range(len(transf_disp.eig_inv)):
            # vec = transf_disp.eig_inv[i]
            # print(np.dot(vec,vec))
            # vec = transf_disp.eig_inv.T[i]
            # print(np.dot(vec,vec))

        # print("Test Harmonic Frequencies:")
        # test_GF = GFMethod(
            # g_init,
            # f_init,
            # self.zmat_obj,
            # self.TED_obj,
            # self.options,
            # self.symm_obj.symtext,
            # cma=cma,
        # )
        # test_GF.run()
        # print(test_GF.L)
        # raise RuntimeError
        
        # This code below is a table of the TED for the final
        # frequencies in the basis of the initial internal coordinates.
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Final TED"))
        print("////////////////////////////////////////////")
        self.TED_obj.run(np.dot(init_GF.L, final_GF.L), final_GF.freq, self.symm_obj.symtext)
        # self.TED_obj.run(final_GF.L, final_GF.freq, self.symm_obj.symtext)

        # This code prints out the frequencies in order of energy as well
        # as the ZPVE in several different units.
        print(
            "Final Harmonic ZPVE in: "
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
        
        # To be implemented: logic for when the final force constants are computed 
        # atop a non-stationary point.

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
            self.options,
        )
        cart_conv.run()

        # if mol2.size != 0:
        #    if self.options.rmsd:
        #        rmsd_geom = RMSD()
        #        rmsd_geom.run(mol1,mol2)

        print("Frequency Shift (cm^-1): ")
        print(final_GF.freq - init_GF.freq)

        # Write a molden file
        molden = MoldenWriter(self.zmat_obj, transf_disp, final_GF.freq)
        molden.run()

        t2 = time.time()
        print("This program took " + str(t2 - t1) + " seconds to run.")

