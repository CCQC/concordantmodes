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

    def __init__(self, options, proj=None, extra_indices=[]):
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

        # Compute the Level B s-vectors
        s_vec = SVectors(
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_b
        )
        if self.options.man_proj:
            proj = self.proj
            np.set_printoptions(precision=4, linewidth=240)
            # print(proj.shape)
            # print(proj)
            s_vec.run(
                self.zmat_obj.cartesians_b,
                True,
                proj=proj,
                second_order=self.options.second_order,
            )
            s_vec.proj = proj
        else:
            s_vec.run(
                self.zmat_obj.cartesians_b,
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

        if os.path.exists(rootdir + "/fc_b.grad"):
            g_read_obj = GrRead("fc_b.grad")
            g_read_obj.run(self.zmat_obj.cartesians_b)
            print(self.zmat_obj.cartesians_b)

        num_deg_free = s_vec.proj.shape[1]
        # Read in FC matrix in cartesians, then convert to internals.
        # Or compute an initial hessian in internal coordinates.
        self.options.init_bool = False
        if os.path.exists(rootdir + "/fc_b.dat"):
            f_read_obj = FcRead("fc_b.dat")
        elif os.path.exists(rootdir + "/FCMFINAL"):
            f_read_obj = FcRead("FCMFINAL")
        else:
            self.options.init_bool = True
            

            # First generate displacements in internal coordinates
            eigs_b = np.eye(len(s_vec.proj.T))
            coord_type = "internal"
            if self.options.cart_fc_b:
                eigs_b = np.eye(len(self.zmat_obj.cartesians_b.flatten()))
                coord_type = "cartesian"
            if not self.options.deriv_level_b:
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
            
            b_disp = TransfDisp(
                s_vec,
                self.zmat_obj,
                eigs_b,
                True,
                self.TED_obj,
                self.options,
                algo.indices,
                symm_obj=self.symm_obj,
                coord_type=coord_type,
                deriv_level=self.options.deriv_level_b,
                cma_level=cma_level
            )
            b_disp.run()

            prog_b = self.options.program_b
            prog_name_b = prog_b.split("@")[0]

            if self.options.gen_disps_b:
                if os.path.exists(rootdir + "/DispsB"):
                    if os.path.exists(rootdir + "/oldDispsB"):
                        shutil.rmtree(rootdir + "/oldDispsB")
                    shutil.copytree(rootdir + "/DispsB", rootdir + "/oldDispsB")
                    shutil.rmtree(rootdir + "/DispsB")
                
                ref_geom_b = b_disp.disp_cart["ref"]

                dir_obj_b = DirectoryTree(
                    prog_name_b,
                    self.zmat_obj,
                    ref_geom_b,
                    cma_level,
                    b_disp.p_disp,
                    b_disp.m_disp,
                    self.options,
                    algo.indices,
                    self.symm_obj,
                    "templateB.dat",
                    "DispsB",
                    deriv_level=self.options.deriv_level_b,
                )
                dir_obj_b.run()

                if not self.options.calc_b:
                    print(
                        "The Level B displacements have been generated, now they must be run locally."
                    )
                    raise RuntimeError
            else:
                if not os.path.exists(rootdir + "/DispsB"):
                    print(
                        "You need to have a DispsB directory already present if you want to proceed under current conditions!"
                    )
                    raise RuntimeError

            os.chdir(rootdir + "/DispsB")

            if self.options.calc_b:
                disp_list = []
                for i in os.listdir(rootdir + "/DispsB"):
                    disp_list.append(i)

                if self.options.cluster != "sapelo":
                    v_template = VulcanTemplate(
                        self.options, len(disp_list), prog_name_b, prog_b
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
                        self.options, len(disp_list), prog_name_b, prog_b
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

            reap_obj_b = Reap(
                self.options,
                len(eigs_b),
                algo.indices,
                self.symm_obj,
                cma_level,
                deriv_level=self.options.deriv_level_b,
            )
            reap_obj_b.run()

            if not self.options.deriv_level_b:
                p_array_b = reap_obj_b.p_en_array
                m_array_b = reap_obj_b.m_en_array
                ref_en_b = reap_obj_b.ref_en
                deriv_level_b = 0
            else:
                cart_p_array_b = reap_obj_b.p_grad_array
                cart_m_array_b = reap_obj_b.m_grad_array
                p_array_b = np.zeros(np.eye(len(eigs_b)).shape)
                m_array_b = np.zeros(np.eye(len(eigs_b)).shape)
                ref_en_b = None

                # Need to convert this array here from cartesians to internals using projected A-tensor
                for i in range(len(algo.indices)):
                    grad_s_vec = SVectors(
                        self.zmat_obj,
                        self.options,
                        self.zmat_obj.variable_dictionary_b,
                    )
                   
                    grad_s_vec.run(b_disp.p_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.TED_obj.proj)
                    p_array_b[i] = np.dot(cart_p_array_b[i].T, A_proj)
                    
                    grad_s_vec.run(b_disp.m_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.TED_obj.proj)
                    m_array_b[i] = np.dot(cart_m_array_b[i].T, A_proj)

                deriv_level_b = 1

            fc_b = ForceConstant(
                b_disp,
                p_array_b,
                m_array_b,
                ref_en_b,
                self.options,
                algo.indices,
                deriv_level=deriv_level_b,
                coord_type_b=coord_type,
                cma_level=cma_level
            )
            fc_b.run()
            print("Computed Force Constants:")
            print(fc_b.FC)
            if self.options.second_order:
                f_conv_obj = FcConv(
                    fc_b.FC,
                    s_vec,
                    self.zmat_obj,
                    "internal",
                    False,
                    self.TED_obj,
                    self.options,
                )
                f_conv_obj.run(grad=fc_b.gradient)
                fc_b.FC = f_conv_obj.F



        # Temporary code to ensure nothing breaks
        self.options.deriv_level_b = 0
        self.options.deriv_level_a = 0

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
            F = fc_b.FC

        if self.options.coords != "ZMAT" and not self.options.init_bool:
            F = np.dot(self.TED_obj.proj.T, np.dot(F, self.TED_obj.proj))
        if self.options.coords != "ZMAT":
            g_mat.G = np.dot(self.TED_obj.proj.T, np.dot(g_mat.G, self.TED_obj.proj))

        self.options.init_bool = False
        # print("F and then G:")
        F[np.abs(F) < self.options.tol] = 0
        g_mat.G[np.abs(g_mat.G) < self.options.tol] = 0
        del_tol = 1.0e-3
        for row in g_mat.G:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row)*del_tol] = 0 
        g_mat.G = g_mat.G.T
        for row in g_mat.G:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row)*del_tol] = 0 
        g_mat.G = g_mat.G.T


        # for col in g_mat.G.T:
            # abs_col = np.abs(col)
            # col[abs_col < np.max(abs_col)*del_tol] = 0 
        for row in F:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row)*del_tol] = 0 
        F = F.T
        for row in F:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row)*del_tol] = 0 
        F = F.T
        # for col in F.T:
            # abs_col = np.abs(col)
            # col[abs_col < np.max(abs_col)*del_tol] = 0 
        # print(F)
        # print(g_mat.G / (5.48579909065 * (10 ** (-4))))
        
        if len(sym_sort) > 1:
            F, g_mat.G = self.symm_obj.GF_sym_sort(F, g_mat, sym_sort)

        # Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
        print("Level B Frequencies:")
        b_GF = GFMethod(
            g_mat.G.copy(),
            F.copy(),
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma="init", # Change to cma_level?
        )
        b_GF.run()

        # For a quick test, to be deleted later
        g_b = g_mat.G.copy()
        f_b = F.copy()

        ted_b = b_GF.ted.TED
        #more sym_sort stuff
        if len(sym_sort):
            self.irreps_b,flat_sym_freqs = self.symm_obj.mode_symmetry_sort(b_GF.ted.TED,sym_sort,b_GF.freq)
            self.ref_b = np.array(flat_sym_freqs)
            #### this block could probably be moved inside the symmetry.py module? 
            flat_sym_modes_b = [
                x
                for xs in self.irreps_b
                for x in xs
            ]
            print(flat_sym_modes_b)
            del_list = []
            # for i in range(len(self.irreps_b)):
                # if len(self.irreps_b[i]) == 1:
                    # del_list.append(self.irreps_b[i][0])
            flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b),del_list)
            ted_b = ted_b.T
            ted_b = ted_b[flat_sym_modes_b]
            ted_b = ted_b.T
            #### end of block that could probably be moved inside the symmetry.py module? 


        # Now for the TED check.
        self.G = np.dot(np.dot(LA.inv(b_GF.L), g_mat.G), LA.inv(b_GF.L).T)
        self.G[np.abs(self.G) < self.options.tol] = 0
        self.F = np.dot(np.dot(b_GF.L.T, F), b_GF.L)
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

        # initial_fc = TED_GF.eig_v
        #eigs = len(TED_GF.S)
        #TODO this needs to be defined with respect to the s_vector number degrees of freedom
        #eigs = len(TED_GF.eig_v)
        #print(f"The eigs {eigs}")
        #print(f"The shape of s_vec {s_vec.proj.shape}")
        #print(stop)


        # Insert statement here for CMA-2, if relevant, to compute level C hessian
        if self.options.off_diag == 2:
            print("We're finally doin it live!")
            # self.extra_indices = []
            if self.options.de_novo_c:
                pass
            else:
                print("No extra computations, running back through level B single points")
                os.chdir("DispsB")
                print(os.getcwd())
                reap_obj_c = Reap(
                    self.options,
                    len(eigs_b),
                    algo.indices,
                    self.symm_obj,
                    "C",
                    deriv_level=self.options.deriv_level_b,
                )
                reap_obj_c.run()
                FC_c = ForceConstant(
                    b_disp,
                    reap_obj_c.p_en_array,
                    reap_obj_c.m_en_array,
                    reap_obj_c.ref_en,
                    self.options,
                    algo.indices,
                    deriv_level=deriv_level_b,
                    coord_type_b=coord_type,
                    cma_level=cma_level
                )
                FC_c.run()
                np.set_printoptions(precision=4, linewidth=240)
                fc_c = FC_c.FC
                fc_c = np.dot(np.dot(b_GF.L.T, fc_c), b_GF.L)
                fc_c[np.abs(fc_c) < self.options.tol] = 0
                print("Level C normal mode Force Constants:")
                print(fc_c)
                xi = copy.deepcopy(fc_c)*0.0
                for i in range(len(xi)):
                    for j in range(i + 1):
                        if i != j:
                            print(i,j)
                            buff = np.abs(fc_c[i, j])
                            xi[i, j] = buff / np.sqrt(np.abs(fc_c[i, i]) * np.abs(fc_c[j, j]))
                            if xi[i, j] > self.options.xi_tol:
                                self.extra_indices.append([j, i])
                                # self.extra_indices.append([j, i])
                # self.extra_indices = self.extra_indices.reshape((-1,2)).astype(int)
                print("CMA-2 extra off-diag indices:")
                print(self.extra_indices)
                # raise RuntimeError





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
            self.zmat_obj, self.options, self.zmat_obj.variable_dictionary_a
        )
        s_vec.run(self.zmat_obj.cartesians_a, False, proj=self.TED_obj.proj)

        # self.options.disp = 0.02

        # print(algo.indices)
        # raise RuntimeError

        transf_disp = TransfDisp(
            s_vec,
            # s_vec.B,
            self.zmat_obj,
            b_GF.L,
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
        prog = self.options.program_a
        progname = prog.split("@")[0]
        if self.options.gen_disps_a:
            if os.path.exists(rootdir + "DispsA"):
                if os.path.exists(rootdir + "/oldDispsA"):
                    shutil.rmtree(rootdir + "/oldDispsA")
                shutil.copytree(rootdir + "/DispsA", rootdir + "/oldDispsA")
                shutil.rmtree(rootdir + "/DispsA")
            
            ref_geom_a = transf_disp.disp_cart["ref"]
            dir_obj_a = DirectoryTree(
                progname,
                self.zmat_obj,
                ref_geom_a,
                cma_level,
                p_disp,
                m_disp,
                self.options,
                algo.indices,
                self.symm_obj,
                "templateA.dat",
                "DispsA",
            )
            dir_obj_a.run()

            if not self.options.calc_a:
                print(
                    "The displacements have been generated, now they must be run locally."
                )
                raise RuntimeError
        else:
            if not os.path.exists(rootdir + "/DispsA"):
                print(
                    "You need to have a DispsA directory already present if you want to proceed under current conditions!"
                )
                raise RuntimeError

        os.chdir(rootdir + "/DispsA")

        if self.options.calc_a:
            disp_list = []
            for i in os.listdir(rootdir + "/DispsA"):
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
        reap_obj_a = Reap(
            self.options,
            num_deg_free,
            algo.indices,
            self.symm_obj,
            cma_level,
        )
        reap_obj_a.run()

        # raise RuntimeError

        p_en_array_a = reap_obj_a.p_en_array
        m_en_array_a = reap_obj_a.m_en_array
        ref_en_a = reap_obj_a.ref_en


        fc_a = ForceConstant(
            transf_disp, p_en_array_a, m_en_array_a, ref_en_a, self.options, algo.indices, cma_level=cma_level
        )
        fc_a.run()
        
        
        np.set_printoptions(precision=4, linewidth=240)
        print("Computed Force Constants:")
        print(fc_a.FC)
        
        self.F = fc_a.FC
        f_b = np.dot(np.dot(LA.inv(transf_disp.eig_inv).T, f_b), LA.inv(transf_disp.eig_inv))
        print("Level B FC in same basis:")
        print(f_b)
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
        # print("Normal flat_sym_modesMode G init:")
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
        a_GF = GFMethod(
            self.G,
            self.F,
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma=cma,
        )
        a_GF.run()

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
        self.TED_obj.run(np.dot(b_GF.L, a_GF.L), a_GF.freq, self.symm_obj.symtext)
        # self.TED_obj.run(final_GF.L, final_GF.freq, self.symm_obj.symtext)

        # This code prints out the frequencies in order of energy as well
        # as the ZPVE in several different units.
        print(
            "Final Harmonic ZPVE in: "
            + "{:6.2f}".format(np.sum(a_GF.freq) / 2)
            + " (cm^-1) "
            + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 349.7550881133)
            + " (kcal mol^-1) "
            + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 219474.6313708)
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
        print(a_GF.freq - b_GF.freq)
        for i in a_GF.freq - b_GF.freq:
            print(i)

        # Write a molden file
        molden = MoldenWriter(self.zmat_obj, transf_disp, a_GF.freq)
        molden.run()

        t2 = time.time()
        print("This program took " + str(t2 - t1) + " seconds to run.")

