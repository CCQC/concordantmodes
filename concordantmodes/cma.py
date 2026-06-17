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

# from concordantmodes.vulcan_template import VulcanTemplate
# from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.zmat import Zmat


class ConcordantModes:
    """
    Driver class for the Concordant Modes Algorithm (CMA).

    This class orchestrates the complete vibrational analysis workflow,
    including:

    - Reading and processing molecular geometries and internal coordinates.
    - Generating symmetry-adapted displacement coordinates via molsym.
    - Constructing internal-coordinate Hessians from finite-difference
      energies or gradients.
    - Performing low-level (Level B) and high-level (Level A) force-constant
      calculations within the CMA framework.
    - Computing G and F matrices and solving the vibrational GF equations.
    - Generating Total Energy Distribution (TED) analyses.
    - Transforming force constants between internal and Cartesian coordinates.
    - Producing harmonic vibrational frequencies, zero-point vibrational
      energies (ZPVE), normal modes, and Molden output files.

    The implementation supports conventional harmonic analyses, projected
    coordinate spaces, symmetry-adapted displacements, reduced-displacement
    CMA calculations, and optional second-order coordinate transformations.

    Parameters
    ----------
    options : object
        User-defined options controlling coordinate generation, electronic
        structure calculations, finite-difference procedures, symmetry
        handling, and output settings.
    proj : numpy.ndarray, optional
        Projection matrix defining the vibrational coordinate space. An empty
        array indicates that the projection will be generated automatically.
    extra_indices : list, optional
        Additional displacement index pairs used for selective off-diagonal
        force-constant evaluation in higher-order CMA procedures.

    Attributes
    ----------
    MDYNE_HART : float
        Conversion factor from Hartrees to mdyne·Å.
    BOHR_ANG : float
        Conversion factor from Bohr to Angstrom.
    proj : numpy.ndarray
        Active projection matrix defining a set of linearly independent internal coordinates.
    zmat_obj : Zmat
        Molecular internal-coordinate representation.
    symm_obj : Symmetry
        Symmetry analysis and displacement-reduction handler.
    s_vec : SVectors
        Internal-coordinate B-matrix and projection-space generator.
    TED_obj : TED
        Total Energy Distribution analysis object.
    disp : TransfDisp
        Coordinate displacement and transformation manager.
    G : numpy.ndarray
        Final transformed G matrix.
    F_a : numpy.ndarray
        High-level (Level A) force-constant matrix.
    F_b : numpy.ndarray
        Low-level (Level B) force-constant matrix.

    Notes
    -----
    The CMA procedure computes a lower-cost reference Hessian (Level B),
    obtains vibrational normal coordinates from a GF analysis, and then
    evaluates selected force constants at a higher level of theory (Level A)
    in the Level B normal coordinate basis. This approach significantly reduces
    the computational cost of high-accuracy vibrational frequency calculations
    while retaining much of the accuracy of a full Level A Hessian.
    """

    def __init__(self, options, proj=np.array([]), extra_indices=[]):
        self.options = options
        # These constants are from:
        # https://physics.nist.gov/cgi-bin/cuu/Value?hr
        # If this link dies, find the new link on NIST
        # for the Hartree to Joule conversion and pop it in there.
        # There is a standard uncertainty of 0.0000000000085 to the MDYNE_HART constant.
        # BOHR_ANG: Standard uncertainty of 0.00000000080
        self.MDYNE_HART = 4.3597447222071
        self.BOHR_ANG = 0.529177210903
        self.proj = proj
        self.extra_indices = extra_indices

    # Change to 'cma' rather than run. We should then be able to introduce a function for higher order
    # force constant computations.
    def run(self, sym_sort=[]):
        t1 = time.time()

        rootdir = os.getcwd()

        # string value to access "state" of CMA program. Are we in level A, B, C, etc...
        cma_level = "B"

        self.sym_sort = sym_sort

        # Parse the output to get all pertinent ZMAT info
        self.zmat_obj = Zmat(self.options)
        self.zmat_obj.run()
        if self.options.geom_check:
            raise RuntimeError

        # Do we want to use molsym_symmetry or "manual" symmetry via sym_sort?
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
            # check if sym_sort object was passed in. If so, intialize sym_sort objects
            if len(self.sym_sort) > 1:
                self.symm_obj.create_flat_sym_sort(self.sym_sort)

        coord_type = "internal"
        if self.options.cart_fc_b:
            coord_type = "cartesian"
        self.F_b, self.grad_b = self.compute_hessian(
            cma_level,
            self.options.deriv_level_b,
            coord_type,
            self.zmat_obj,
            self.options,
            rootdir,
            self.zmat_obj.cartesians_b,
            proj=self.proj,
            off_diag=0,
            prog=self.options.program_b,
            gen_disps=self.options.gen_disps_b,
            calc=self.options.calc_b,
        )

        g_mat = GMatrix(self.zmat_obj, self.s_vec, self.options, proj=self.proj)
        g_mat.run()
        G = g_mat.G.copy()

        self.options.init_bool = False

        if len(self.sym_sort) > 1:
            F, g_mat.G = self.symm_obj.GF_sym_sort(self.F_b, g_mat, self.sym_sort)

        # Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
        print("Level B Frequencies:")
        b_GF = GFMethod(
            g_mat.G.copy(),
            self.F_b.copy(),
            self.zmat_obj,
            self.TED_obj,
            self.options,
            symtext=self.symm_obj.symtext,
        )
        b_GF.run()

        ted_b = b_GF.ted.TED
        # more sym_sort stuff
        if len(self.sym_sort):
            self.irreps_b, flat_sym_freqs = self.symm_obj.mode_symmetry_sort(
                ted_b, self.sym_sort, b_GF.freq
            )
            self.ref_b = np.array(flat_sym_freqs)
            #### this block could probably be moved inside the symmetry.py module?
            flat_sym_modes_b = [x for xs in self.irreps_b for x in xs]
            print(flat_sym_modes_b)
            del_list = []
            flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b), del_list)
            ted_b = ted_b.T
            ted_b = ted_b[flat_sym_modes_b]
            ted_b = ted_b.T
            #### end of block that could probably be moved inside the symmetry.py module?

        self.F_b = np.dot(np.dot(b_GF.L.T, self.F_b), b_GF.L)
        # Now for the TED check.
        if self.options.ted_check:
            self.G = np.dot(np.dot(LA.inv(b_GF.L), g_mat.G), LA.inv(b_GF.L).T)
            self.G[np.abs(self.G) < self.options.tol] = 0
            self.F = np.dot(np.dot(b_GF.L.T, self.F_b), b_GF.L)
            self.F[np.abs(self.F) < self.options.tol] = 0

            print("TED Frequencies: Degeneracy x Irrep")
            TED_GF = GFMethod(
                self.G,
                self.F,
                self.zmat_obj,
                self.TED_obj,
                self.options,
                self.symm_obj.symtext,
                cma=False,
            )
            TED_GF.run()

        # Insert statement here for CMA-2, if relevant, to compute level C hessian
        # This code is incomplete ***.
        if self.options.off_diag == 2:
            # self.extra_indices = []
            if self.options.de_novo_C:
                pass
            else:
                print(
                    "No extra computations, running back through level B single points"
                )
                os.chdir("DispsB")
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
                    deriv_level=self.options.deriv_level_b,
                    coord_type=coord_type,
                    cma_level=cma_level,
                )
                FC_c.run()
                np.set_printoptions(precision=4, linewidth=240)
                fc_c = FC_c.FC
                fc_c = np.dot(np.dot(b_GF.L.T, fc_c), b_GF.L)
                fc_c[np.abs(fc_c) < self.options.tol] = 0
                xi = copy.deepcopy(fc_c) * 0.0
                print("xi values:")
                for i in range(len(xi)):
                    for j in range(i + 1):
                        if i != j:
                            print(i, j)
                            buff = np.abs(fc_c[i, j])
                            xi[i, j] = buff / np.sqrt(
                                np.abs(fc_c[i, i]) * np.abs(fc_c[j, j])
                            )
                            print(i, j)
                            print(xi[i, j])
                            if xi[i, j] > self.options.xi_tol:
                                self.extra_indices.append([j, i])
                print("CMA-2 extra off-diag indices:")
                print(len(self.extra_indices))
                print(self.extra_indices)
        # elif self.options.off_diag == '1':
        # self.extra_indices = self.od_inds

        # Can we generalize compute_hessian to run this too?
        # Now switch state to cma_level = "A"
        cma_level = "A"

        fc = np.array([])
        if self.options.reduced_disp:
            fc = self.F_b
        self.F_a, self.grad_a = self.compute_hessian(
            cma_level,
            self.options.deriv_level_a,
            "internal",
            self.zmat_obj,
            self.options,
            rootdir,
            self.zmat_obj.cartesians_a,
            proj=self.s_vec.proj,
            off_diag=self.options.off_diag,
            prog=self.options.program_a,
            gen_disps=self.options.gen_disps_a,
            calc=self.options.calc_a,
            eigs=b_GF.L,
            fc=fc,
        )

        # Recompute the G-matrix with the new geometry, and then transform
        # the G-matrix using the lower level of theory eigenvalue matrix.
        # This will not fully diagonalize the G-matrix if a different
        # geometry is used between the two.
        g_mat = GMatrix(self.zmat_obj, self.s_vec, self.options, proj=self.proj)
        g_mat.run()
        np.set_printoptions(precision=7, linewidth=240)

        self.G = g_mat.G

        self.G = np.dot(np.dot(self.disp.eig_inv, self.G), self.disp.eig_inv.T)
        self.G[np.abs(self.G) < self.options.tol] = 0

        if self.options.benchmark_full:
            cma = True
        else:
            cma = False

        # Final GF Matrix run
        print("Final Harmonic Frequencies:")
        a_GF = GFMethod(
            self.G,
            self.F_a,
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
            cma=cma,
        )
        a_GF.run()

        # This code below is a table of the TED for the final
        # frequencies in the basis of the initial internal coordinates.
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format(" Final TED"))
        print("////////////////////////////////////////////")
        self.TED_obj.run(np.dot(b_GF.L, a_GF.L), a_GF.freq, self.symm_obj.symtext)

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

        self.F_a = np.dot(np.dot(self.disp.eig_inv.T, self.F_a), self.disp.eig_inv)
        self.gradient = np.dot(self.grad_a, self.disp.eig_inv)

        cart_conv = FcConv(
            self.F_a,
            self.s_vec,
            self.zmat_obj,
            "cartesian",
            True,
            self.proj,
            self.options,
        )
        if self.options.second_order:
            cart_conv.run(grad=self.gradient)
            self.grad_cart = cart_conv.grad
        else:
            cart_conv.run()

        self.F_cart = cart_conv.F

        print("Frequency Shift (cm^-1): ")
        print(a_GF.freq - b_GF.freq)
        for i in a_GF.freq - b_GF.freq:
            print(i)
        print("RMSD Freq Shift (cm^-1): ")
        print(np.sqrt(np.mean((a_GF.freq - b_GF.freq) ** 2)))
        print("MAX Freq Shift (cm^-1): ")
        print(np.max(np.abs(a_GF.freq - b_GF.freq)))

        # Write a molden file
        molden = MoldenWriter(self.zmat_obj, self.disp, a_GF.freq)
        molden.run()

        t2 = time.time()
        print("This program took " + str(t2 - t1) + " seconds to run.")

    # It's time to make this a real driver. Begin by creating a procedure for computing an arbitrary hessian
    # whether it's a full hessian, variable derivative levels, variable CMA_levels, variable coordinate types.
    def compute_hessian(
        self,
        cma_level,
        deriv_level,
        coord_type,
        zmat,
        options,
        rootdir,
        ref_carts,
        proj=np.array([]),
        off_diag=None,
        prog=None,
        gen_disps=True,
        calc=True,
        eigs=None,
        fc=np.array([]),
    ):
        self.s_vec = SVectors(zmat, options)
        b_proj = False
        if cma_level == "B" and not options.man_proj:
            b_proj = True
        self.s_vec.run(
            ref_carts,
            b_proj,
            proj=proj,
            second_order=options.second_order,
        )
        self.proj = self.s_vec.proj

        self.TED_obj = TED(self.proj, zmat, options)

        if os.path.exists(rootdir + "/fc_" + cma_level.lower() + ".grad"):
            g_read_obj = GrRead("fc_" + cma_level.lower() + ".grad")
            # Need to pass in general carts here.
            g_read_obj.run(ref_carts)

        num_deg_free = self.proj.shape[1]
        if deriv_level and self.options.cart_fc_b:
            num_deg_free = len(ref_carts.flatten())
        options.init_bool = False
        cart_fc = False
        if os.path.exists(rootdir + "/fc_" + cma_level.lower() + ".dat"):
            f_read_obj = FcRead("fc_" + cma_level.lower() + ".dat")
            cart_fc = True

            f_read_obj.run()
            f_conv_obj = FcConv(
                f_read_obj.fc_mat,
                self.s_vec,
                zmat,
                "internal",
                False,
                self.TED_obj,
                options,
            )
            if self.options.second_order:
                f_conv_obj.run(grad=g_read_obj.cart_grad)
            else:
                f_conv_obj.run()
            F = f_conv_obj.F
        else:
            options.init_bool = True

            # First generate displacements in internal coordinates
            if cma_level == "B":
                eigs = np.eye(len(self.proj.T))
                if self.options.cart_fc_b:
                    eigs = np.eye(len(ref_carts.flatten()))
                    cart_fc = True

            algo = Algorithm(
                num_deg_free,
                cma_level,
                options,
                proj_irreps=self.symm_obj.proj_irreps,
            )
            algo.run()
            print("Pre sym indices:")
            print(algo.indices)
            # raise RuntimeError
            if (
                not options.deriv_level_b
                and not options.molsym_symmetry
                and cma_level == "B"
            ):
                # Sym_sort doesn't seem to be working
                print("symmetric displacements:")
                if len(self.sym_sort) > 1:
                    algo.indices = self.symm_obj.create_sym_sort_disps(
                        self.sym_sort, algo.indices
                    )
            else:
                self.symm_obj.indices_by_irrep = algo.indices_by_irrep
            print("Post sym indices:")
            print(algo.indices)
            if cma_level == "A" and len(self.extra_indices):
                algo.indices += self.extra_indices
            self.disp = TransfDisp(
                self.s_vec,
                zmat,
                eigs,
                True,
                self.proj,
                options,
                algo.indices,
                symm_obj=self.symm_obj,
                coord_type=coord_type,
                deriv_level=deriv_level,
                cma_level=cma_level,
            )
            self.disp.run(fc=fc)

            prog_name = prog.split("@")[0]

            if gen_disps:
                ref_geom = self.disp.disp_cart["ref"]

                dir_obj = DirectoryTree(
                    prog_name,
                    zmat,
                    ref_geom,
                    cma_level,
                    self.disp.p_disp,
                    self.disp.m_disp,
                    options,
                    algo.indices,
                    self.symm_obj,
                    "template" + cma_level.upper() + ".dat",
                    "Disps" + cma_level.upper(),
                    deriv_level=deriv_level,
                )
                dir_obj.run()

                if not calc:
                    print(
                        # "The Level B displacements have been generated, now they must be run locally."
                        "The Level "
                        + cma_level.upper()
                        + " displacements have been generated, now they must be run locally."
                    )
                    raise RuntimeError
            else:
                if not os.path.exists(rootdir + "/Disps" + cma_level.upper()):
                    print(
                        "You need to have a Disps"
                        + cma_level.upper()
                        + " directory already present if you want to proceed under current conditions!"
                    )
                    raise RuntimeError

            if calc:
                sub = Submit(options, cma_level, rootdir, prog_name, prog)
                sub.run()

            os.chdir(rootdir + "/Disps" + cma_level.upper())

            reap_obj = Reap(
                options,
                len(eigs),
                algo.indices,
                self.symm_obj,
                cma_level,
                deriv_level=deriv_level,
                disp=self.disp,
                proj=self.proj,
                zmat=zmat,
            )
            reap_obj.run()
            os.chdir(rootdir)
            # can this be folded into reap obj?
            if not deriv_level:
                ref_en = reap_obj.ref_en
                p_array = reap_obj.p_en_array
                m_array = reap_obj.m_en_array
            else:
                p_array = reap_obj.p_grad_array
                m_array = reap_obj.m_grad_array
                ref_en = None

            fc = ForceConstant(
                self.disp,
                p_array,
                m_array,
                ref_en,
                options,
                algo.indices,
                deriv_level=deriv_level,
                coord_type=coord_type,
                cma_level=cma_level,
                gradient=reap_obj.ref_grad,
            )
            fc.run()

            if options.second_order and cart_fc:
                f_conv_obj = FcConv(
                    fc.FC,
                    self.s_vec,
                    zmat,
                    "internal",
                    False,
                    self.proj,
                    options,
                )
                f_conv_obj.run(grad=fc.gradient)
                fc.FC = f_conv_obj.F

            F = fc.FC

        F[np.abs(F) < self.options.tol] = 0
        del_tol = 1.0e-3
        for row in F:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row) * del_tol] = 0
        F = F.T
        for row in F:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row) * del_tol] = 0
        F = F.T
        return F, fc.gradient
