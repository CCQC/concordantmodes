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
from concordantmodes.zmat import Zmat
from concordantmodes import fc_files, programs, workflow
from concordantmodes.errors import CMAError, CMAWaiting
from concordantmodes.log import debug_print, info_print, setup_logging
import logging

logger = logging.getLogger(__name__)


class ConcordantModes:
    """
    Driver class for the Concordant Mode Approach (CMA).

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
        # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0|search_for=bohr
        # If this link dies, find the new link on NIST
        # for the Hartree to Joule conversion and pop it in there.
        # There is a standard uncertainty of 0.0000000000048 to the MDYNE_HART constant.
        # BOHR_ANG: Standard uncertainty of 0.00000000082
        self.MDYNE_HART = 4.3597447222060
        self.BOHR_ANG = 0.529177210544
        self.proj = proj
        self.extra_indices = extra_indices

    def run(self, sym_sort=[]):
        """Run (or continue) the CMA calculation in the current directory.

        CMA works out from the files on disk what still needs doing: it
        writes displacement inputs that do not exist yet, submits or waits
        for calculations that are not finished (depending on ``cluster``),
        and reads the results once every calculation of a level is done.
        Run it again after your calculations finish; completed work is
        never redone.

        Returns ``"finished"`` when the final frequencies were computed, or
        ``"waiting"`` when calculations still have to run (the message
        explains what to run).
        """
        self.options.validate()
        verbosity = self.options.verbosity
        if self.options.geom_check and verbosity == "normal":
            verbosity = "debug"  # geom_check exists to show the parsed structure
        setup_logging(verbosity, self.options.log_file)
        rootdir = os.getcwd()
        logger.info("Concordant Modes run in %s", rootdir)
        changed = self.options.changed()
        if changed:
            logger.debug("Options set: %s", changed)
        for name in self.options.no_effect_set():
            logger.warning("Option '%s' is set but currently has no effect.", name)
        try:
            workflow.preflight(self.options, rootdir)
            self._run(sym_sort)
            self.status = "finished"
        except CMAWaiting as waiting:
            os.chdir(rootdir)
            # Shown even with verbosity="quiet": it says what to do next.
            logger.warning("%s", waiting)
            self.status = "waiting"
        except CMAError as error:
            os.chdir(rootdir)
            logger.error("ERROR: %s", error)
            error.logged = True
            raise
        return self.status

    def _run(self, sym_sort=[]):
        t1 = time.time()

        rootdir = os.getcwd()

        # string value to access "state" of CMA program. Are we in level A, B, C, etc...
        cma_level = "B"

        self.sym_sort = sym_sort

        # Parse the output to get all pertinent ZMAT info
        self.zmat_obj = Zmat(self.options)
        self.zmat_obj.run()
        if self.options.geom_check:
            raise CMAWaiting(
                "geom_check is set: the zmat file was read (details above). Remove "
                "geom_check to run the calculation."
            )
        if self.options.off_diag == 2:
            raise CMAError(
                "CMA-2 (off_diag = 2) is not finished in this version: the level-C "
                "block in cma.py uses variables that are never defined (eigs_b, "
                "algo, b_disp). Use off_diag = 0."
            )

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
        )

        g_mat = GMatrix(self.zmat_obj, self.s_vec, self.options, proj=self.proj)
        g_mat.run()
        G = g_mat.G.copy()
        debug_print(logger, G.shape)
        debug_print(logger, self.F_b.shape)

        self.options.init_bool = False

        if len(self.sym_sort) > 1:
            self.F_b, G = self.symm_obj.GF_sym_sort(self.F_b, G, self.sym_sort)

        # Run the GF matrix method with the internal F-Matrix and computed G-Matrix!
        info_print(logger, "Level B harmonic frequencies (cm^-1):")
        b_GF = GFMethod(
            G.copy(),
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
            debug_print(logger, flat_sym_modes_b)
            del_list = []
            flat_sym_modes_b = np.delete(np.array(flat_sym_modes_b), del_list)
            ted_b = ted_b.T
            ted_b = ted_b[flat_sym_modes_b]
            ted_b = ted_b.T
            #### end of block that could probably be moved inside the symmetry.py module?

        self.disp = TransfDisp(
            None,
            self.zmat_obj,
            None,
            None,
            None,
            self.options,
            None,
        )
        eig_inv = self.disp._build_eig_inv(b_GF.L)

        # Use inv of normalized eig_inv instead of b_GF.L?
        # self.F_b = np.dot(np.dot(b_GF.L.T, self.F_b), b_GF.L)
        self.F_b = np.dot(np.dot(inv(eig_inv).T, self.F_b), inv(eig_inv))
        # Now for the TED check. The statement above could be problematic
        # for the TED check. Might want to fold it into an "else"
        # statement to the TED check. Also, might want to use
        # normalized eig_inv for TED check, it's just a scale factor
        # to the diagonal G-matrix anyways...
        if self.options.ted_check:
            self.G = np.dot(np.dot(LA.inv(b_GF.L), G), LA.inv(b_GF.L).T)
            self.G[np.abs(self.G) < self.options.tol] = 0
            self.F = np.dot(np.dot(b_GF.L.T, self.F_b), b_GF.L)
            self.F[np.abs(self.F) < self.options.tol] = 0

            debug_print(logger, "TED Frequencies: Degeneracy x Irrep")
            TED_GF = GFMethod(
                self.G,
                self.F,
                self.zmat_obj,
                self.TED_obj,
                self.options,
                self.symm_obj.symtext,
            )
            TED_GF.run()

        # Insert statement here for CMA-2, if relevant, to compute level C hessian
        # This code is incomplete ***.
        if self.options.off_diag == 2:
            # self.extra_indices = []
            if self.options.de_novo_C:
                pass
            else:
                debug_print(
                    logger,
                    "No extra computations, running back through level B single points",
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
                debug_print(logger, "xi values:")
                for i in range(len(xi)):
                    for j in range(i + 1):
                        if i != j:
                            debug_print(logger, i, j)
                            buff = np.abs(fc_c[i, j])
                            xi[i, j] = buff / np.sqrt(
                                np.abs(fc_c[i, i]) * np.abs(fc_c[j, j])
                            )
                            debug_print(logger, i, j)
                            debug_print(logger, xi[i, j])
                            if xi[i, j] > self.options.xi_tol:
                                self.extra_indices.append([j, i])
                debug_print(logger, "CMA-2 extra off-diag indices:")
                debug_print(logger, len(self.extra_indices))
                debug_print(logger, self.extra_indices)
        # elif self.options.off_diag == '1':
        # self.extra_indices = self.od_inds

        # Can we generalize compute_hessian to run this too?
        # Now switch state to cma_level = "A"
        cma_level = "A"
        self.options.cart_fc_b = False

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
            eigs=b_GF.L,
            fc=fc,
        )
        # debug_print(logger, self.F_a)

        # Recompute the G-matrix with the new geometry, and then transform
        # the G-matrix using the lower level of theory eigenvalue matrix.
        # This will not fully diagonalize the G-matrix if a different
        # geometry is used between the two.
        g_mat = GMatrix(self.zmat_obj, self.s_vec, self.options, proj=self.proj)
        g_mat.run()
        np.set_printoptions(precision=7, linewidth=240)

        self.G = g_mat.G

        if len(self.sym_sort) > 1:
            _, self.G = self.symm_obj.GF_sym_sort(
                np.zeros(self.F_b.shape), self.G, self.sym_sort
            )

        self.G = np.dot(np.dot(eig_inv, self.G), eig_inv.T)
        self.G[np.abs(self.G) < self.options.tol] = 0

        np.set_printoptions(precision=7, edgeitems=60, linewidth=10000)
        debug_print(logger, "Normal Mode G")
        debug_print(logger, self.G)
        debug_print(logger, "Diag F")
        debug_print(logger, self.F_a)

        # Final GF Matrix run
        info_print(logger, "\nFinal (CMA) harmonic frequencies (cm^-1):")
        a_GF = GFMethod(
            self.G,
            self.F_a,
            self.zmat_obj,
            self.TED_obj,
            self.options,
            self.symm_obj.symtext,
        )
        a_GF.run()

        # This code below is a table of the TED for the final
        # frequencies in the basis of the initial internal coordinates.
        info_print(
            logger,
            "\nFinal total energy distribution (TED, % per internal coordinate):",
        )
        self.TED_obj.run(
            np.dot(b_GF.L, a_GF.L), a_GF.freq, self.symm_obj.symtext, show=True
        )

        # This code prints out the frequencies in order of energy as well
        # as the ZPVE in several different units.
        info_print(
            logger,
            "Final Harmonic ZPVE in: "
            + "{:6.2f}".format(np.sum(a_GF.freq) / 2)
            + " (cm^-1) "
            + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 349.7550881133)
            + " (kcal mol^-1) "
            + "{:6.2f}".format(0.5 * np.sum(a_GF.freq) / 219474.6313708)
            + " (hartrees) ",
        )

        shift = a_GF.freq - b_GF.freq
        info_print(logger, "\nFrequency shift from level B to CMA (cm^-1):")
        info_print(logger, "  mode      level B        CMA      shift")
        for i, (fb, fa) in enumerate(zip(b_GF.freq, a_GF.freq)):
            info_print(
                logger,
                "  {:4d} {:12.2f} {:10.2f} {:10.2f}".format(i + 1, fb, fa, fa - fb),
            )
        info_print(
            logger,
            "  RMS shift {:.2f}, largest |shift| {:.2f}".format(
                np.sqrt(np.mean(shift**2)), np.max(np.abs(shift))
            ),
        )

        # Write a molden file
        if self.options.write_molden:
            molden = MoldenWriter(self.zmat_obj, self.disp, a_GF.freq)
            molden.run()

        # This code converts the force constants back into cartesian
        # coordinates and writes force_constants/fc_cart_a.dat (and
        # fc_cart_a.grad with second_order), in CFOUR FCMFINAL format.

        self.F_a = np.dot(np.dot(eig_inv.T, self.F_a), eig_inv)
        if np.size(self.grad_a) == eig_inv.shape[0]:
            self.gradient = np.dot(self.grad_a, eig_inv)
        else:
            # fc_int_a.dat from an older version, saved without fc_int_a.grad.
            self.gradient = np.zeros(eig_inv.shape[1])
            if self.options.second_order:
                logger.warning(
                    "second_order is set but no level-A gradient was found "
                    "(force_constants/fc_int_a.grad); a zero gradient is used for the "
                    "Cartesian force constants. Delete force_constants/fc_int_a.dat "
                    "to recompute both from DispsA."
                )

        cart_conv = FcConv(
            self.F_a,
            self.s_vec,
            self.zmat_obj,
            "cartesian",
            True,
            self.proj,
            self.options,
            out_dir=os.path.join(rootdir, fc_files.FC_DIR),
        )
        if self.options.second_order:
            cart_conv.run(grad=self.gradient)
            self.grad_cart = cart_conv.grad
        else:
            cart_conv.run()

        self.F_cart = cart_conv.F

        t2 = time.time()
        info_print(
            logger,
            "\nFinished in {:.1f} s. Cartesian force constants: force_constants/fc_cart_a.dat; "
            "{}full log: {}".format(
                t2 - t1,
                "normal modes: MOLDEN; " if self.options.write_molden else "",
                self.options.log_file or "(disabled)",
            ),
        )

    # It's time to make this a real driver. Begin by creating a procedure for computing an arbitrary hessian
    # whether it's a full hessian, variable derivative levels, variable CMA_levels, variable coordinate types.
    def _build_displacements(
        self,
        cma_level,
        coord_type,
        zmat,
        options,
        ref_carts,
        num_deg_free,
        eigs,
        deriv_level,
        fc,
    ):
        """Choose the displacement indices and build the displaced geometries
        (``self.disp``) for one level. Nothing is written to disk."""
        # First generate displacements in internal coordinates
        if cma_level.upper() == "B":
            eigs = np.eye(len(self.proj.T))
            if self.options.cart_fc_b:
                eigs = np.eye(len(ref_carts.flatten()))
                # cart_fc = True

        algo = Algorithm(
            num_deg_free,
            cma_level,
            options,
            proj_irreps=self.symm_obj.proj_irreps,
        )
        algo.run()
        debug_print(logger, "Pre sym indices:")
        debug_print(logger, len(algo.indices))
        debug_print(logger, algo.indices)
        if (
            not options.deriv_level_b
            and not options.molsym_symmetry
            and cma_level == "B"
        ):
            if len(self.sym_sort) > 1 and coord_type.lower() == "internal":
                debug_print(logger, "symmetric displacements:")
                algo.indices = self.symm_obj.create_sym_sort_disps(
                    self.sym_sort, algo.indices
                )
        else:
            self.symm_obj.indices_by_irrep = algo.indices_by_irrep

        debug_print(logger, "Post sym indices:")
        debug_print(logger, len(algo.indices))
        debug_print(logger, algo.indices)

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
        return algo

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
            proj=self.proj,
            second_order=options.second_order,
        )

        if self.options.molsym_symmetry:
            self.symm_obj.make_proj(self.s_vec)
            self.s_vec.proj = copy.deepcopy(self.symm_obj.salc_proj)

        self.proj = self.s_vec.proj

        self.TED_obj = TED(self.proj, zmat, options)

        num_deg_free = self.proj.shape[1]
        options.init_bool = False
        cart_fc = False
        if coord_type.lower() == "cartesian":
            self.options.cart_fc_b = True
            num_deg_free = len(ref_carts.flatten())
            cart_fc = True
            suff = "cart_"
        elif coord_type.lower() == "internal":
            suff = "int_"
        is_cart = suff == "cart_"
        # If the force-constant file exists it is used and Disps<L> is ignored.
        # Delete force_constants/fc_int_a.dat to re-read DispsA.
        fc_path = fc_files.find(rootdir, fc_files.fc_name(cma_level, is_cart))
        grad_path = fc_files.find(rootdir, fc_files.fc_name(cma_level, is_cart, "grad"))
        if fc_path:
            logger.info(
                "Level %s: reading force constants from %s",
                cma_level,
                os.path.relpath(fc_path, rootdir),
            )
            f_read_obj = FcRead(fc_path)

            f_read_obj.run()
            grad = np.array([])
            if not is_cart and grad_path:
                # Gradient in the same coordinates as the force constants.
                grad = fc_files.read_vector(grad_path)
            if self.options.cart_fc_b:
                f_conv_obj = FcConv(
                    f_read_obj.fc_mat,
                    self.s_vec,
                    zmat,
                    "internal",
                    False,
                    self.proj,
                    options,
                )
                if self.options.second_order and grad_path:
                    g_read_obj = GrRead(grad_path)
                    # Need to pass in general carts here.
                    g_read_obj.run(ref_carts)
                    f_conv_obj.run(grad=g_read_obj.cart_grad)
                    grad = f_conv_obj.grad
                else:
                    f_conv_obj.run()
                F = f_conv_obj.F
            else:
                F = f_read_obj.fc_mat
            if cma_level.upper() == "A" and options.write_molden:
                # The MOLDEN file is drawn from the level-A displacements, so
                # rebuild them in memory (no inputs are written).
                options.init_bool = True
                self._build_displacements(
                    cma_level,
                    coord_type,
                    zmat,
                    options,
                    ref_carts,
                    num_deg_free,
                    eigs,
                    deriv_level,
                    fc,
                )
        else:
            options.init_bool = True

            algo = self._build_displacements(
                cma_level,
                coord_type,
                zmat,
                options,
                ref_carts,
                num_deg_free,
                eigs,
                deriv_level,
                fc,
            )

            prog_name = programs.program_name(prog)
            disps_name = "Disps" + cma_level.upper()

            def make_tree(dir_name):
                return DirectoryTree(
                    prog_name,
                    zmat,
                    self.disp.disp_cart["ref"],
                    cma_level,
                    self.disp.p_disp,
                    self.disp.m_disp,
                    options,
                    algo.indices,
                    self.symm_obj,
                    "template" + cma_level.upper() + ".dat",
                    dir_name,
                    deriv_level=deriv_level,
                )

            reap_obj = Reap(
                options,
                num_deg_free,
                algo.indices,
                self.symm_obj,
                cma_level,
                deriv_level=deriv_level,
                disp=self.disp,
                proj=self.proj,
                zmat=zmat,
            )

            # Write inputs if needed, check what is finished, and submit or
            # stop until the remaining calculations are done.
            workflow.prepare_level(
                options,
                rootdir,
                cma_level,
                disps_name,
                make_tree,
                reap_obj,
                prog_name,
                prog,
            )

            os.chdir(rootdir + "/" + disps_name)
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

            f_conv_obj = FcConv(
                fc.FC,
                self.s_vec,
                zmat,
                "internal",
                False,
                self.proj,
                options,
            )
            f_conv_obj.N = len(fc.FC)
            if options.second_order and self.options.cart_fc_b:
                f_conv_obj.print_const(
                    fc_name=fc_files.output_path(
                        rootdir, fc_files.fc_name(cma_level, True)
                    )
                )
                fc_files.write_vector(
                    fc_files.output_path(
                        rootdir, fc_files.fc_name(cma_level, True, "grad")
                    ),
                    fc.gradient,
                )

                f_conv_obj.run(grad=fc.gradient)
                fc.FC = f_conv_obj.F
            elif not self.options.cart_fc_b:
                f_conv_obj.print_const(
                    fc_name=fc_files.output_path(rootdir, fc_files.fc_name(cma_level))
                )
                # Saved so a later run that reads the force-constant file gets
                # the same gradient (needed for second_order).
                if np.size(fc.gradient):
                    fc_files.write_vector(
                        fc_files.output_path(
                            rootdir, fc_files.fc_name(cma_level, ext="grad")
                        ),
                        fc.gradient,
                    )

            F = fc.FC
            grad = fc.gradient

        F[np.abs(F) < self.options.tol] = 0
        return F, grad
