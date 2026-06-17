import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class FcConv:
    """
    Transform force-constant matrices between Cartesian and internal
    coordinate representations.

    This class performs the coordinate transformation of harmonic force
    constants using the Wilson B-matrix formalism. Transformations may be
    performed in either direction:

    - Cartesian → Internal coordinates
    - Internal → Cartesian coordinates

    The class also supports projection into a reduced vibrational coordinate
    space and optional second-order corrections required for non-stationary
    geometries when Cartesian Hessians are transformed into internal
    coordinates.

    For Cartesian-to-internal transformations, the generalized inverse of
    the Wilson B-matrix is constructed using

    Aᵀ = G⁻¹B

    where

    G = BBᵀ

    is the Wilson G-matrix.

    Optional gradient-dependent corrections based on second derivatives of
    the internal coordinates (B² tensors) can be included when transforming
    Hessians obtained at non-stationary points on the potential energy
    surface.

    Parameters
    ----------
    fc_mat : ndarray
        Input force-constant matrix. The interpretation depends on the
        selected coordinate system:

        - Cartesian Hessian when ``coord="internal"``
        - Internal-coordinate force constant matrix when
          ``coord="cartesian"``

    s_vec : SVectors
        Internal-coordinate object containing Wilson B-matrices and,
        optionally, second-order B tensors.

    zmat : Zmat
        Molecular coordinate representation.

    coord : {"internal", "cartesian"}
        Desired output coordinate system.

        - ``"internal"`` transforms Cartesian force constants into
          internal coordinates.
        - ``"cartesian"`` transforms internal force constants into
          Cartesian coordinates.

    print_f : bool
        If True, write transformed force constants to disk.

    proj : ndarray
        Projection matrix defining a reduced vibrational coordinate basis.
        An empty array indicates that the full internal-coordinate basis
        should be used.

    options : object
        User options controlling units, second-order transformations,
        and output generation.

    Attributes
    ----------
    F : ndarray
        Transformed force-constant matrix.

    A_T : ndarray
        Transpose of the generalized inverse transformation matrix used
        for Cartesian-to-internal transformations.

    grad : ndarray
        Transformed gradient vector when second-order corrections are
        applied.

    v_q : ndarray
        Internal-coordinate gradient vector used in second-order
        corrections.

    MDYNE_HART : float
        Conversion factor between Hartree and mdyne·Å.

    BOHR_ANG : float
        Conversion factor between Bohr and Angstrom.

    Notes
    -----
    The Cartesian-to-internal transformation is performed as

    F_int = Aᵀ F_cart A

    while the reverse transformation is

    F_cart = Bᵀ F_int B

    where B is the Wilson B-matrix and A is its generalized inverse.

    When second-order transformations are enabled, the force constants are
    corrected using second derivatives of the internal coordinates stored
    in the B² tensor. This correction is required whenever the reference
    geometry is not a stationary point and gradient contributions do not
    vanish.

    The transformed force constants may optionally be written in a format
    compatible with CFOUR-style force-constant files.
    """
    # The constants are from:
    # https://physics.nist.gov/cgi-bin/cuu/Value?hr
    # If this link dies, find the new link on NIST
    # for the Hartree to Joule conversion and pop
    # it in there.
    # MDYNE_HART: Standard uncertainty of 0.0000000000085
    # BOHR_ANG: Standard uncertainty of 0.00000000080

    def __init__(self, fc_mat, s_vec, zmat, coord, print_f, proj, options):
        self.coord = coord
        self.F = fc_mat
        self.print_f = print_f
        self.s_vec = s_vec
        self.proj = proj
        self.zmat = zmat
        self.options = options

        self.MDYNE_HART = 4.3597447222071
        self.BOHR_ANG = 0.529177210903

    def run(self, grad=np.array([])):
        """
        Perform the force-constant coordinate transformation.

        Depending on the value of ``self.coord``, transforms the input
        force-constant matrix between Cartesian and internal coordinate
        representations. If gradient information is supplied and second-order
        transformations are enabled, gradient-dependent corrections are
        included.

        Parameters
        ----------
        grad : ndarray, optional
            Gradient vector at the reference geometry. Required for
            second-order transformations involving non-stationary geometries.

        Returns
        -------
        None

        Notes
        -----
        Results are stored in the instance attribute ``F``. If second-order
        corrections are applied, the transformed gradient is stored in
        ``grad`` and the correction term is computed from the second-order B tensor.
        """
        # First construct the transpose of the A matrix.
        if self.coord.lower() == "internal":
            # The cartesian force constants must be in units of Hartree/bohr^2.
            if not len(self.proj):
                B = self.s_vec.B
            else:
                B = np.dot(self.proj.T, self.s_vec.B)
            G = np.dot(B, B.T)
            self.A_T = np.dot(LA.inv(G), B)
            if self.options.units == "MdyneAng":
                self.F /= self.BOHR_ANG
                self.F *= self.MDYNE_HART
            self.F = np.einsum("pi,rj,ij->pr", self.A_T, self.A_T, self.F)
            # Non-stationary, gradient correction to internal coordinate force constants
            V2 = self.F.copy() * 0
            if len(grad) and self.options.second_order and self.options.cart_fc_b:
                # Note: I may want to use projected A_T to reduce the size of the
                # internal coordinate basis, thus speeding up the computation.
                # The basis will eventually need to be projected anyways.
                np.set_printoptions(precision=6, linewidth=240)
                self.v_q = np.dot(self.A_T, grad)
                if not len(self.proj):
                    B2 = self.s_vec.B2
                else:
                    B2 = np.einsum("rp,pij->rij", self.proj.T, self.s_vec.B2)
                C2 = np.einsum("rij,pi->rpj", B2, self.A_T)
                C2 = np.einsum("rpj,qj->rpq", C2, self.A_T)
                V2 = np.einsum("q,qpr->pr", self.v_q, C2)

                grad = np.dot(grad, self.A_T.T)

            self.F -= V2

            if self.print_f:
                self.print_const(fc_name="fc_int.dat", grad=grad)
        elif self.coord.lower() == "cartesian":

            if not len(self.proj):
                B = self.s_vec.B
            else:
                B = np.dot(self.proj.T, self.s_vec.B)

            self.F = np.einsum("pi,rj,pr->ij", B, B, self.F)

            V2 = self.F.copy() * 0

            if len(grad) and self.options.second_order:
                if not len(self.proj):
                    B2 = self.s_vec.B2
                else:
                    B2 = np.einsum("rp,pij->rij", self.proj.T, self.s_vec.B2)
                V2 = np.einsum("rij,r->ij", B2, grad)

                grad = np.dot(grad, B)
                self.grad = grad

            self.F += V2

            if self.print_f:
                self.print_const(grad=grad)

    def print_const(self, fc_name="fc_a.dat", grad=np.array([])):
        """
        Write transformed force constants and gradients to disk.

        The force-constant matrix is flattened and written in a format
        compatible with CFOUR-style force-constant files. If a gradient is
        provided, it is written to a separate gradient file.

        Parameters
        ----------
        fc_name : str, optional
            Output force-constant filename.

        grad : ndarray, optional
            Gradient vector to be written alongside the force constants.

        Returns
        -------
        None
        """
        self.N = len(self.F)
        fc_output = ""
        fc_output += "{:5d}{:5d}\n".format(len(self.zmat.atom_list), self.N)
        print("print_const has run")
        self.F_print = self.F.copy()
        self.F_print = self.F_print.flatten()
        for i in range(len(self.F_print) // 3):
            fc_output += "{:20.10f}".format(self.F_print[3 * i])
            fc_output += "{:20.10f}".format(self.F_print[3 * i + 1])
            fc_output += "{:20.10f}".format(self.F_print[3 * i + 2])
            fc_output += "\n"
        if len(self.F_print) % 3:
            for i in range(len(self.F_print) % 3):
                fc_output += "{:20.10f}".format(
                    self.F_print[3 * (len(self.F_print) // 3) + i]
                )
            fc_output += "\n"
        with open(fc_name, "w+") as file:
            file.write(fc_output)

        if len(grad):
            g_print = grad
            gr_output = ""
            for i in range(len(g_print) // 3):
                gr_output += "{:20.10f}".format(g_print[3 * i])
                gr_output += "{:20.10f}".format(g_print[3 * i + 1])
                gr_output += "{:20.10f}".format(g_print[3 * i + 2])
                gr_output += "\n"
            if len(g_print) % 3:
                for i in range(len(g_print) % 3):
                    gr_output += "{:20.10f}".format(
                        g_print[3 * (len(g_print) // 3) + i]
                    )
                fc_output += "\n"
            # Introduce grad_name
            with open("fc_a.grad", "w+") as file:
                file.write(gr_output)
