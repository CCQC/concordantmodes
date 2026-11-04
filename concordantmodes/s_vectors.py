import numpy as np
from concordantmodes.ted import TED
from concordantmodes.int2cart import Int2Cart
from numpy import linalg as LA
import logging
from concordantmodes.log import debug_print, info_print

logger = logging.getLogger(__name__)


class SVectors:
    """
    Construct first- and second-order Wilson B-tensors (s-vectors) for a
    molecular internal coordinate system.

    This class evaluates the derivatives of internal coordinates with respect
    to Cartesian coordinates and assembles them into the Wilson B-matrix used
    throughout the GF method, coordinate transformations, force-constant
    conversions, and vibrational analyses. Supported internal coordinate types
    include bond stretches, angle bends, torsions, out-of-plane bends, linear
    bends, and specialized linear bending coordinates (LinX and LinY).

    For an internal coordinate q_k and atom a, the corresponding s-vector is

        s_a^k = (∂q_k/∂x_a, ∂q_k/∂y_a, ∂q_k/∂z_a)

    and each row of the Wilson B-matrix is formed by concatenating the
    s-vectors for all atoms associated with a given internal coordinate.

    The class can optionally:

    * Generate a projection matrix defining a linearly independent internal
      coordinate basis using singular value decomposition (SVD).
    * Compute numerical second-order B-tensors (B2) through finite
      differentiation of first-order B-matrices.
    * Support projected internal coordinate spaces for delocalized or
      non-redundant coordinate analyses.

    Parameters
    ----------
    zmat : Zmat
        Molecular coordinate definition containing atom labels, masses,
        Cartesian coordinates, and internal coordinate index lists
        (bonds, angles, torsions, out-of-plane bends, etc.).
    options : Options
        Program options controlling coordinate generation, projection,
        displacement magnitudes, numerical tolerances, and second-order
        derivative calculations.

    Attributes
    ----------
    bond_indices : list
        Bond stretch coordinate definitions.
    angle_indices : list
        Angle bend coordinate definitions.
    torsion_indices : list
        Dihedral angle coordinate definitions.
    oop_indices : list
        Out-of-plane bending coordinate definitions.
    lin_indices : list
        Linear bend coordinate definitions.
    linx_indices : list
        Linear bend X-component coordinate definitions.
    liny_indices : list
        Linear bend Y-component coordinate definitions.
    B : ndarray
        First-order Wilson B-matrix with dimensions
        (N_internal, 3N_atoms).
    B2 : ndarray, optional
        Numerical second-order Wilson B-tensor with dimensions
        (N_internal, 3N_atoms, 3N_atoms).
    proj : ndarray
        Projection matrix mapping redundant internal coordinates to a
        linearly independent coordinate space.
    s_2center_dict : dict
        First-order s-vectors for two-center coordinates (bonds).
    s_3center_dict : dict
        First-order s-vectors for three-center coordinates (angles).
    s_4center_dict : dict
        First-order s-vectors for four-center coordinates (torsions,
        out-of-plane bends, and linear bends).
    s_ncenter_dict : dict
        Reserved storage for generalized multi-center coordinates.

    Notes
    -----
    The analytical expressions for bond, angle, torsional, out-of-plane,
    and linear-coordinate derivatives are based on the Wilson GF formalism.
    Out-of-plane derivative expressions follow the conventions described in
    Wilson, Decius, and Cross, *Molecular Vibrations*, with the atom ordering
    adapted to the coordinate conventions used by Concordant Modes.

    When `second_order=True`, numerical differentiation of displaced
    B-matrices is used to generate the second-order tensor B2, which is
    required for non-stationary force constant transformations and
    higher-order coordinate transformations.

    References
    ----------
    Wilson, E. B.; Decius, J. C.; Cross, P. C.
    *Molecular Vibrations: The Theory of Infrared and Raman Vibrational
    Spectra*; Dover Publications, 1980.
    """

    def __init__(self, zmat, options):
        self.s_2center_dict = {}
        self.s_3center_dict = {}
        self.s_4center_dict = {}
        self.s_ncenter_dict = {}
        self.bond_indices = zmat.bond_indices
        self.angle_indices = zmat.angle_indices
        self.torsion_indices = zmat.torsion_indices
        self.oop_indices = zmat.oop_indices
        self.lin_indices = zmat.lin_indices
        self.linx_indices = zmat.linx_indices
        self.liny_indices = zmat.liny_indices
        self.options = options
        self.zmat = zmat

    def run(self, carts, B_proj, proj=np.array([]), second_order=False):
        self.proj = proj
        # Initialize the cartesian coordinates
        self.carts = carts
        self.int2cart = Int2Cart(self.zmat)
        # So, first things first I'll have to code up the
        # proper equations for the first order B-Tensors
        # for:
        # Bonds,
        # Angles,
        # Torsions,
        # Out-of-plane bending,
        # Linear Bends
        # LinX Bends
        # LinY Bends

        self.B = self._build_B()

        # if len(self.proj):
        # debug_print(logger, "Redundancy Check:")
        # # B_projected = np.dot(self.proj.T,self.B)
        # B_projected = self.B
        # proj, eigs, _ = LA.svd(B_projected)
        # for i in eigs:
        # debug_print(logger, i)
        # debug_print(logger, proj[:,-1])
        # raise RuntimeError

        tol = self.options.red_tol
        # Now we acquire a linearly independant set of internal coordinates from the diagonalized
        # BB^T Matrix
        if not self.options.man_proj:
            if self.options.coords.upper() != "ZMAT":
                # The SVD is only needed when the projection is being
                # (re)built. Skipping it otherwise avoids a full SVD on every
                # displaced geometry (second-order B, gradient conversion).
                if B_proj:
                    proj, eigs, _ = LA.svd(self.B)
                    # proj[np.abs(proj) < tol] = 0
                    debug_print(logger, "proj singular values:")
                    debug_print(logger, eigs)
                    proj_array = np.array(np.where(np.abs(eigs) > tol))
                    self.proj = proj.T[: len(proj_array[0])]
                    self.proj = self.proj.T
            else:
                self.proj = np.eye(len(self.B))

        # self.proj may be used to transform from the full set of internal
        # coords to projected internal coords. self.proj.T may be used
        # to transform from the projected set to the full set of internal
        # coords.

        # Beware! The projected B matrix cannot be psuedo inverted to form
        # the A-matrix. You lose information.

        # Run numerical second order B-tensor here.
        if second_order:
            B = self.B.copy()
            Proj = self.proj.copy()
            self.B2 = self.second_order_B()
            self.B2[np.abs(self.B2) < 1e-14] = 0
            np.set_printoptions(precision=2, linewidth=2000)

            # Average the off-diagonal elements to mitigate the numerical errors
            self.B2 = self.B2.astype(float)
            self.B2 = 0.5 * (self.B2 + np.swapaxes(self.B2, 1, 2))
            self.proj = Proj
            self.B = B

    # ------------------------------------------------------------------
    # Vectorized B-matrix construction
    # ------------------------------------------------------------------
    # Every coordinate of a given type is evaluated at once on (n, 3)
    # arrays. Scalars (distances, angles) are kept as (n, 1) columns so
    # the element-wise helpers above (compute_BEND, compute_TORS1, ...)
    # broadcast unchanged; only the helpers that use np.dot or LA.norm
    # need the row-wise versions below.

    # (type name, key prefix, s-vector dict, number of centers that get
    # an s-vector). Order matches the row order of the B-matrix.
    _COORD_TYPES = (
        ("bond", "B", "s_2center_dict", 2),
        ("angle", "A", "s_3center_dict", 3),
        ("torsion", "D", "s_4center_dict", 4),
        ("oop", "O", "s_4center_dict", 4),
        ("lin", "L", "s_4center_dict", 3),
        ("linx", "Lx", "s_4center_dict", 4),
        ("liny", "Ly", "s_4center_dict", 4),
    )

    def _index_cache(self):
        """
        Build (once) padded atom-index and count arrays for each coordinate
        type. A center is either one atom or a group of atoms (centroid).

        For a type with n coordinates, k centers and at most g atoms per
        center: idx (n, k, g) holds 0-based atom indices (padding repeats
        index 0), mask (n, k, g) is 1 for real entries and 0 for padding,
        and count (n, k, 1) is the number of atoms in each center.
        """
        cache = getattr(self, "_idx_cache", None)
        if cache is not None:
            return cache
        cache = {}
        for name, _, _, _ in self._COORD_TYPES:
            coords = getattr(self, name + "_indices")
            if not len(coords):
                continue
            centers = [
                [
                    [int(a) - 1 for a in g] if np.ndim(g) else [int(g) - 1]
                    for g in groups
                ]
                for groups in coords
            ]
            n = len(centers)
            k = len(centers[0])
            g_max = max(len(c) for groups in centers for c in groups)
            idx = np.zeros((n, k, g_max), dtype=int)
            mask = np.zeros((n, k, g_max))
            for i, groups in enumerate(centers):
                for j, c in enumerate(groups):
                    idx[i, j, : len(c)] = c
                    mask[i, j, : len(c)] = 1.0
            count = mask.sum(axis=2, keepdims=True)
            cache[name] = (idx, mask, count)
        self._idx_cache = cache
        return cache

    def _batch_points(self, idx, mask, count):
        """Center positions, one (n, 3) array per center."""
        carts = np.asarray(self.carts, dtype=float)
        if idx.shape[2] == 1:
            pts = carts[idx[:, :, 0]]
        else:
            # Centroid of each group (padding masked out).
            pts = (carts[idx] * mask[..., None]).sum(axis=2) / count
        return [pts[:, j] for j in range(pts.shape[1])]

    def _build_B(self):
        """
        Assemble the Wilson B-matrix, vectorized over all coordinates of
        each type. Produces the same rows, in the same order, as
        ``_build_B_reference``.
        """
        n_atoms = len(self.carts)
        cache = self._index_cache()
        blocks = []
        for name, prefix, dict_name, n_used in self._COORD_TYPES:
            if name not in cache:
                continue
            idx, mask, count = cache[name]
            values = getattr(self, "_svec_" + name)(
                *self._batch_points(idx, mask, count)
            )
            n = idx.shape[0]
            block = np.zeros((n, n_atoms, 3))
            rows = np.arange(n)
            for j in range(n_used):
                if idx.shape[2] == 1:
                    np.add.at(block, (rows, idx[:, j, 0]), values[j])
                else:
                    val = values[j] / count[:, j]
                    for g in range(idx.shape[2]):
                        np.add.at(
                            block,
                            (rows, idx[:, j, g]),
                            val * mask[:, j, g][:, None],
                        )
            store = getattr(self, dict_name)
            for i in range(n):
                store[prefix + str(i + 1)] = block[i]
            blocks.append(block.reshape(n, 3 * n_atoms))
        if not blocks:
            return np.array([])
        return np.concatenate(blocks, axis=0)

    # Row-wise versions of the helpers that use np.dot / LA.norm.
    @staticmethod
    def _rdot(a, b):
        return np.einsum("ij,ij->i", a, b)[:, None]

    @staticmethod
    def _rnorm(v):
        return np.sqrt(np.einsum("ij,ij->i", v, v))[:, None]

    def _r(self, x1, x2):
        return self._rnorm(x1 - x2)

    def _e(self, x1, x2):
        return (x1 - x2) / self._r(x1, x2)

    def _phi(self, e_1, e_2):
        return np.arccos(np.clip(self._rdot(e_1, e_2), -1, 1))

    def _calc_OOP_v(self, x1, x2, x3, x4):
        e1 = (x1 - x2) / self._rnorm(x1 - x2)
        e3 = (x3 - x2) / self._rnorm(x3 - x2)
        e4 = (x4 - x2) / self._rnorm(x4 - x2)
        phi = np.arccos(self._rdot(e3, e4))
        return np.arcsin(self._rdot(np.cross(e3, e4) / np.sin(phi), e1))

    def _calc_Lin_v(self, x1, x2, x3, x4):
        e1 = (x1 - x2) / self._rnorm(x1 - x2)
        e3 = (x3 - x2) / self._rnorm(x3 - x2)
        e4 = (x4 - x2) / self._rnorm(x4 - x2)
        return np.arcsin(self._rdot(e4, np.cross(e3, e1)))

    def _calc_alpha_x_v(self, e_1, e_2, e_3):
        theta_1 = self._phi(e_1, -e_2)
        theta_2 = self._phi(e_2, -e_3)
        a = np.cos(theta_1) * np.cos(theta_2)
        b = self._rdot(e_1, -e_3)
        return (a + b) / np.sin(theta_1)

    def _calc_alpha_y_v(self, e_1, e_2, e_3):
        theta = self._phi(e_1, -e_2)
        s = self._rdot(e_1, np.cross(e_2, -e_3))
        return s / np.sin(theta)

    def _compute_LIN_v(self, e_1, e_2, e_3, r, theta):
        c = np.cross(e_3, e_2)
        return (c - self._rdot(e_1, c) * e_1) / (np.cos(theta) * r)

    # One method per coordinate type: center positions in, list of
    # per-center s-vectors out (same formulas as _build_B_reference).
    def _svec_bond(self, x1, x2):
        e = self._e(x1, x2)
        return [e, -e]

    def _svec_angle(self, x1, x2, x3):
        r1 = self._r(x1, x2)
        r2 = self._r(x2, x3)
        e1 = self._e(x1, x2)
        e2 = self._e(x3, x2)
        phi = self._phi(e1, e2)
        a1 = self.compute_BEND(e1, e2, phi, r1)
        a3 = self.compute_BEND(e2, e1, phi, r2)
        return [a1, -a1 - a3, a3]

    def _svec_torsion(self, x1, x2, x3, x4):
        r_1 = self._r(x1, x2)
        r_2 = self._r(x2, x3)
        r_3 = self._r(x3, x4)
        e_1 = self._e(x1, x2)
        e_2 = self._e(x2, x3)
        e_3 = self._e(x3, x4)
        phi_1 = self._phi(e_1, -e_2)
        phi_2 = self._phi(e_2, -e_3)
        t1 = self.compute_TORS1(e_1, -e_2, phi_1, r_1)
        t4 = self.compute_TORS1(-e_3, e_2, phi_2, r_3)
        t2 = self.compute_TORS2(e_1, -e_2, -e_3, phi_1, phi_2, r_1, r_2)
        return [t1, t2, -t1 - t2 - t4, t4]

    def _svec_oop(self, x1, x2, x3, x4):
        r_1 = self._r(x1, x2)
        r_2 = self._r(x3, x2)
        r_3 = self._r(x4, x2)
        e_1 = self._e(x1, x2)
        e_2 = self._e(x3, x2)
        e_3 = self._e(x4, x2)
        phi = self._phi(e_2, e_3)
        theta = self._calc_OOP_v(x1, x2, x3, x4)
        o1 = self.compute_OOP1(e_1, e_2, e_3, r_1, theta, phi)
        o3 = self.compute_OOP2(e_1, e_2, e_3, r_2, theta, phi)
        o4 = self.compute_OOP2(-e_1, e_3, e_2, r_3, theta, phi)
        return [o1, -o1 - o3 - o4, o3, o4]

    def _svec_lin(self, x1, x2, x3, x4):
        r_1 = self._r(x1, x2)
        r_2 = self._r(x3, x2)
        e_1 = self._e(x1, x2)
        e_2 = self._e(x3, x2)
        e_3 = self._e(x4, x2)
        theta = self._calc_Lin_v(x1, x2, x3, x4)
        l1 = self._compute_LIN_v(e_1, e_2, e_3, r_1, theta)
        l3 = self._compute_LIN_v(e_2, e_3, e_1, r_2, theta)
        return [l1, -l1 - l3, l3]

    def _svec_linx(self, x1, x2, x3, x4):
        r_1 = self._r(x2, x1)
        r_2 = self._r(x3, x2)
        r_3 = self._r(x4, x3)
        e_1 = self._e(x1, x2)
        e_2 = self._e(x2, x3)
        e_3 = self._e(x3, x4)
        ax = self._calc_alpha_x_v(e_1, e_2, e_3)
        phi_1 = self._phi(e_1, -e_2)
        phi_2 = self._phi(e_2, -e_3)
        lx1 = self.compute_LINX1(e_1, e_2, e_3, r_1, phi_1, phi_2, ax)
        lx2 = self.compute_LINX2(e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax)
        lx4 = self.compute_LINX4(e_1, e_2, e_3, r_3, phi_1, ax)
        return [lx1, lx2, -lx1 - lx2 - lx4, lx4]

    def _svec_liny(self, x1, x2, x3, x4):
        r_1 = self._r(x2, x1)
        r_2 = self._r(x3, x2)
        r_3 = self._r(x4, x3)
        e_1 = self._e(x1, x2)
        e_2 = self._e(x2, x3)
        e_3 = self._e(x3, x4)
        ax = self._calc_alpha_x_v(e_1, e_2, e_3)
        ay = self._calc_alpha_y_v(e_1, e_2, e_3)
        phi_1 = self._phi(e_1, -e_2)
        phi_2 = self._phi(e_2, -e_3)
        ly1 = self.compute_LINY1(e_1, e_2, e_3, r_1, phi_1, ay)
        ly2 = self.compute_LINY2(e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax, ay)
        ly4 = self.compute_LINY4(e_1, e_2, e_3, r_3, phi_1, ay)
        return [ly1, ly2, -ly1 - ly2 - ly4, ly4]

    def _build_B_reference(self):
        """
        Original per-coordinate (loop) construction of the B-matrix.

        Kept as a slow reference implementation: the test suite checks that
        the vectorized ``_build_B`` reproduces it. Not used in production.
        """
        # First, bonds.
        for i, groups in enumerate(self.bond_indices):
            x1, x2 = self._get_points(groups)

            e = self.compute_e(x1, x2)
            values = [e, -e]

            key = f"B{i+1}"
            self._build_s_vector(key, groups, values, self.s_2center_dict)

        # Next, angles.
        for i, groups in enumerate(self.angle_indices):
            x1, x2, x3 = self._get_points(groups)

            r1 = self.compute_r(x1, x2)
            r2 = self.compute_r(x2, x3)
            e1 = self.compute_e(x1, x2)
            e2 = self.compute_e(x3, x2)

            phi = self.compute_phi(e1, e2)

            a1 = self.compute_BEND(e1, e2, phi, r1)
            a3 = self.compute_BEND(e2, e1, phi, r2)
            a2 = -a1 - a3

            values = [a1, a2, a3]

            key = f"A{i+1}"
            self._build_s_vector(key, groups, values, self.s_3center_dict)

        # Next, torsions.
        for i, groups in enumerate(self.torsion_indices):
            x1, x2, x3, x4 = self._get_points(groups)

            r_1 = self.compute_r(x1, x2)
            r_2 = self.compute_r(x2, x3)
            r_3 = self.compute_r(x3, x4)
            e_1 = self.compute_e(x1, x2)
            e_2 = self.compute_e(x2, x3)
            e_3 = self.compute_e(x3, x4)
            phi_1 = self.compute_phi(e_1, -e_2)
            phi_2 = self.compute_phi(e_2, -e_3)

            t1 = self.compute_TORS1(e_1, -e_2, phi_1, r_1)
            t4 = self.compute_TORS1(-e_3, e_2, phi_2, r_3)
            t2 = self.compute_TORS2(e_1, -e_2, -e_3, phi_1, phi_2, r_1, r_2)
            t3 = -t1 - t2 - t4

            values = [t1, t2, t3, t4]

            key = f"D{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # Now, out of plane bending.
        for i, groups in enumerate(self.oop_indices):
            x1, x2, x3, x4 = self._get_points(groups)

            r_1 = self.compute_r(x1, x2)
            r_2 = self.compute_r(x3, x2)
            r_3 = self.compute_r(x4, x2)
            e_1 = self.compute_e(x1, x2)
            e_2 = self.compute_e(x3, x2)
            e_3 = self.compute_e(x4, x2)
            phi = self.compute_phi(e_2, e_3)
            theta = self.calc_OOP(x1, x2, x3, x4)

            o1 = self.compute_OOP1(e_1, e_2, e_3, r_1, theta, phi)
            o3 = self.compute_OOP2(e_1, e_2, e_3, r_2, theta, phi)
            o4 = self.compute_OOP2(-e_1, e_3, e_2, r_3, theta, phi)
            o2 = -o1 - o3 - o4

            values = [o1, o2, o3, o4]

            key = f"O{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # Linear bending.
        for i, groups in enumerate(self.lin_indices):
            x1, x2, x3, x4 = self._get_points(groups)

            r_1 = self.compute_r(x1, x2)
            r_2 = self.compute_r(x3, x2)
            r_3 = self.compute_r(x4, x2)
            e_1 = self.compute_e(x1, x2)
            e_2 = self.compute_e(x3, x2)
            e_3 = self.compute_e(x4, x2)
            theta = self.calc_Lin(x1, x2, x3, x4)

            l1 = self.compute_LIN(e_1, e_2, e_3, r_1, theta)
            l3 = self.compute_LIN(e_2, e_3, e_1, r_2, theta)
            l2 = -l1 - l3

            values = [l1, l2, l3]

            key = f"L{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # LinX bending.
        for i, groups in enumerate(self.linx_indices):
            x1, x2, x3, x4 = self._get_points(groups)

            r_1 = self.compute_r(x2, x1)
            r_2 = self.compute_r(x3, x2)
            r_3 = self.compute_r(x4, x3)
            e_1 = self.compute_e(x1, x2)
            e_2 = self.compute_e(x2, x3)
            e_3 = self.compute_e(x3, x4)
            ax = self.calc_alpha_x(e_1, e_2, e_3)
            phi_1 = self.compute_phi(e_1, -e_2)
            phi_2 = self.compute_phi(e_2, -e_3)

            lx1 = self.compute_LINX1(e_1, e_2, e_3, r_1, phi_1, phi_2, ax)
            lx2 = self.compute_LINX2(e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax)
            lx4 = self.compute_LINX4(e_1, e_2, e_3, r_3, phi_1, ax)
            lx3 = -lx1 - lx2 - lx4

            values = [lx1, lx2, lx3, lx4]

            key = f"Lx{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # LinY bending.
        for i, groups in enumerate(self.liny_indices):
            x1, x2, x3, x4 = self._get_points(groups)

            r_1 = self.compute_r(x2, x1)
            r_2 = self.compute_r(x3, x2)
            r_3 = self.compute_r(x4, x3)
            e_1 = self.compute_e(x1, x2)
            e_2 = self.compute_e(x2, x3)
            e_3 = self.compute_e(x3, x4)
            ax = self.calc_alpha_x(e_1, e_2, e_3)
            ay = self.calc_alpha_y(e_1, e_2, e_3)
            phi_1 = self.compute_phi(e_1, -e_2)
            phi_2 = self.compute_phi(e_2, -e_3)

            ly1 = self.compute_LINY1(e_1, e_2, e_3, r_1, phi_1, ay)
            ly2 = self.compute_LINY2(e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax, ay)
            ly4 = self.compute_LINY4(e_1, e_2, e_3, r_3, phi_1, ay)
            ly3 = -ly1 - ly2 - ly4

            values = [ly1, ly2, ly3, ly4]

            key = f"Ly{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # The last step will be to concatenate all of the s-vectors into a single B-tensor.
        rows = []

        rows += [v.flatten() for v in self.s_2center_dict.values()]
        rows += [v.flatten() for v in self.s_3center_dict.values()]
        rows += [v.flatten() for v in self.s_4center_dict.values()]

        return np.array(rows)

    def compute_STRE(self, x1, x2):
        s = (x1 - x2) / self.compute_r(x1, x2)
        return s

    def compute_BEND(self, e_1, e_2, phi, r):
        s = (e_1 * np.cos(phi) - e_2) / (r * np.sin(phi))
        return s

    def compute_TORS1(self, e_1, e_2, phi, r):
        s = np.cross(e_1, e_2) / (r * np.sin(phi) ** 2)
        return s

    def compute_TORS2(self, e_1, e_2, e_3, phi_1, phi_2, r_1, r_2):
        s = ((r_2 - r_1 * np.cos(phi_1)) / (r_1 * r_2 * np.sin(phi_1) ** 2)) * np.cross(
            e_2, e_1
        ) + (np.cos(phi_2) / (r_2 * np.sin(phi_2) ** 2)) * np.cross(-e_2, e_3)
        return s

    # See Wilson, Decius, and Cross' "Molecular Vibrations" page 60 for the
    # OOP1 and OOP2 formulae.

    # The indices from the textbook have been rearranged in this
    # implementation as follows,
    # 1-->1
    # 2-->3
    # 3-->4
    # 4-->2.

    def compute_OOP1(self, e_1, e_2, e_3, r, theta, phi):
        s = (
            np.cross(e_2, e_3) / (np.cos(theta) * np.sin(phi)) - np.tan(theta) * e_1
        ) / r
        return s

    def compute_OOP2(self, e_1, e_2, e_3, r, theta, phi):
        s = (
            np.cross(e_3, e_1) / (np.cos(theta) * np.sin(phi))
            - (np.tan(theta) * (e_2 - np.cos(phi) * e_3)) / (np.sin(phi) ** 2)
        ) / r
        return s

    def compute_LIN(self, e_1, e_2, e_3, r, theta):
        s = (np.cross(e_3, e_2) - (np.dot(e_1, np.cross(e_3, e_2))) * e_1) / (
            np.cos(theta) * r
        )
        return s

    def compute_LINX1(self, e_1, e_2, e_3, r, phi_1, phi_2, ax):
        denom = r * np.sin(phi_1)
        a = -ax * e_1 / np.sin(phi_1)
        b = -e_2 * (ax / np.tan(phi_1) + np.cos(phi_2))
        s = a + b - e_3
        s /= denom
        return s

    def compute_LINX2(self, e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax):
        cos_1 = np.cos(phi_1)
        sin_1 = np.sin(phi_1)
        denom = r_1 * r_2 * sin_1
        piece_1 = ax * (r_2 - r_1 * cos_1) / sin_1
        a = e_1 * (piece_1 - r_1 * np.cos(phi_2))
        piece_1 = ax * (r_1 * cos_1 - r_2) / np.tan(phi_1)
        piece_2 = np.cos(phi_2) * (2 * r_1 * cos_1 - r_2)
        b = -e_2 * (piece_1 + piece_2)
        c = e_3 * (r_2 - r_1 * cos_1)
        s = (a + b + c) / denom
        return s

    def compute_LINX4(self, e_1, e_2, e_3, r, phi, ax):
        a = e_1 / (r * np.sin(phi))
        b = e_2 / (r * np.tan(phi))
        c = ax * e_3 / r
        s = a + b + c
        return s

    def compute_LINY1(self, e_1, e_2, e_3, r, phi, ay):
        denom = r * np.sin(phi)
        a = ay * (e_1 + np.cos(phi) * e_2) / np.sin(phi)
        b = np.cross(-e_2, -e_3)
        s = -(a + b) / denom
        return s

    def compute_LINY2(self, e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax, ay):
        sin_1 = np.sin(phi_1)
        cos_1 = np.cos(phi_1)
        denom = r_1 * r_2 * sin_1
        a = r_2 * ay * (e_1 + cos_1 * e_2) / sin_1
        b = r_1 * (ax / np.tan(phi_1) + np.cos(phi_2)) * np.cross(e_1, -e_2)
        c = r_2 * np.cross(-e_2, -e_3)
        s = (a + b + c) / denom
        return s

    def compute_LINY4(self, e_1, e_2, e_3, r, phi, ay):
        s = -(np.cross(e_1, -e_2) / np.sin(phi) - e_3 * ay) / r
        return s

    def calc_OOP(self, x1, x2, x3, x4):
        """
        This function will compute an out of plane angle between one bond
        and a plane formed by 3 other atoms. See page 58 of Molecular
        vibrations by Wilson, Decius, and Cross for more info. However,
        the indices have been rearranged as follows,
        1-->1
        2-->3
        3-->4
        4-->2.
        """
        e1 = (x1 - x2) / LA.norm(x1 - x2)
        e3 = (x3 - x2) / LA.norm(x3 - x2)
        e4 = (x4 - x2) / LA.norm(x4 - x2)
        phi = np.arccos(np.dot(e3, e4))
        theta = np.arcsin(np.dot(np.cross(e3, e4) / np.sin(phi), e1))
        return theta

    def calc_Lin(self, x1, x2, x3, x4):
        e1 = (x1 - x2) / LA.norm(x1 - x2)
        e3 = (x3 - x2) / LA.norm(x3 - x2)
        e4 = (x4 - x2) / LA.norm(x4 - x2)
        theta = np.arcsin(np.dot(e4, np.cross(e3, e1)))
        return theta

    def compute_e(self, x1, x2):
        r = self.compute_r(x1, x2)
        e = (x1 - x2) / r
        return e

    def compute_r(self, x1, x2):
        r = LA.norm(x1 - x2)
        return r

    def compute_phi(self, e_1, e_2):
        p = np.dot(e_1, e_2)
        if p > 1:
            p = 1
        if p < -1:
            p = -1
        phi = np.arccos(p)
        return phi

    def calc_alpha_x(self, e_1, e_2, e_3):
        theta_1 = self.compute_phi(e_1, -e_2)
        theta_2 = self.compute_phi(e_2, -e_3)
        a = np.cos(theta_1) * np.cos(theta_2)
        b = np.dot(e_1, -e_3)
        ax = (a + b) / np.sin(theta_1)
        return ax

    def calc_alpha_y(self, e_1, e_2, e_3):
        theta = self.compute_phi(e_1, -e_2)
        s = np.dot(e_1, np.cross(e_2, -e_3))
        ay = s / np.sin(theta)
        return ay

    def calc_r_com(self, cart1, cart2, mass1, mass2):
        rc1 = self.int2cart.COM(cart1, mass1)
        rc2 = self.int2cart.COM(cart2, mass2)
        debug_print(logger, rc1)
        debug_print(logger, rc2)
        rc = rc1 - rc2
        rc = LA.norm(rc)
        debug_print(logger, rc)
        return rc

    def num_differentiate(self, B_list_p, B_list_m):
        disp_size = self.options.disp_b

        # Numerical first derivative
        B = (B_list_p - B_list_m) / (2 * disp_size)

        return B

    def second_order_B(self):
        # Set some initial necessary variables
        B_buff = self.B.copy()
        TED_obj = TED(np.eye(len(self.B)), self.zmat, self.options)

        # Initialize then generate the internal coordinate displacements
        from concordantmodes.transf_disp import TransfDisp

        B_disp = TransfDisp(
            self,
            self.zmat,
            np.eye(len(self.B)),
            True,
            TED_obj,
            self.options,
            np.arange(len(self.B)),
            coord_type="cartesian",
            deriv_level=1,
        )

        B_disp.run()
        # Collect displaced B-matrices in lists and stack once at the end
        # (np.append in a loop copies the whole array every iteration).
        self.run(B_disp.p_disp[0], False, proj=self.proj, second_order=False)
        B_list_p = [self.B.copy()]
        self.run(B_disp.m_disp[0], False, proj=self.proj, second_order=False)
        B_list_m = [self.B.copy()]

        for i in range(len(B_disp.p_disp) - 1):
            self.run(B_disp.p_disp[i + 1], False, second_order=False)
            B_list_p.append(self.B.copy())
            self.run(B_disp.m_disp[i + 1], False, second_order=False)
            B_list_m.append(self.B.copy())

        B_list_p = np.stack(B_list_p).astype(float)
        B_list_m = np.stack(B_list_m).astype(float)

        # And differentiate
        B2 = self.num_differentiate(B_list_p, B_list_m)

        # Put B-tensor in canonical order and average out the numerical fuzz
        B2 = np.swapaxes(B2, 0, 1)
        B2 = (B2 + np.swapaxes(B2, 1, 2)) / 2

        return B2

    def _get_points(self, groups):
        """Return averaged coordinates for grouped or single indices."""
        pts = []
        for g in groups:
            if np.ndim(g):  # calculates centroid. May need logic for COM.
                coords = np.mean([self.carts[int(i) - 1] for i in g], axis=0)
            else:
                coords = self.carts[int(g) - 1]
            pts.append(coords)
        return pts

    def _distribute(self, arr, groups, values):
        """Distribute vector contributions across atoms."""
        for group, val in zip(groups, values):
            if np.ndim(group):
                for i in group:
                    arr[int(i) - 1] += val / len(group)
            else:
                arr[int(group) - 1] = val

    def _build_s_vector(self, key, groups, values, store):
        arr = np.zeros_like(self.carts)
        self._distribute(arr, groups, values)
        store[key] = arr
