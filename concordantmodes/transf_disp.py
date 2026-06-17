import numpy as np
import copy
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from . import masses
from concordantmodes.s_vectors import SVectors as s_vec

np.set_printoptions(precision=8, linewidth=240)


class TransfDisp:
    """
    Generate Cartesian displacement geometries from internal-coordinate or
    Cartesian-coordinate perturbations for finite-difference derivative
    calculations.

    This class performs iterative transformations between normal-coordinate
    displacements and Cartesian coordinates using Wilson's B-matrix formalism.
    It constructs the generalized inverse transformation matrix (A-matrix),
    generates displaced molecular geometries, and optionally includes
    second-order corrections through numerical second derivatives of the
    B-matrix.

    The transformation is based on:

        A = B⁺ P

    where B⁺ is the pseudoinverse of the B-matrix and P is the projection
    matrix defining the nonredundant internal coordinate space. The resulting
    A-matrix is subsequently transformed into the normal-coordinate basis to
    produce Cartesian displacements corresponding to specified vibrational
    normal mode displacements.

    The class supports:

    * Internal-coordinate displacements in normal-coordinate space.
    * Cartesian-coordinate finite-difference displacements.
    * First-derivative (gradient) displacement generation.
    * Second-derivative (Hessian) displacement generation.
    * Reduced or scaled displacement schemes.
    * Optional second-order coordinate transformations using A₂ tensors.
    * Tight iterative coordinate transformations with B-matrix updates.
    * Symmetry-adapted displacement generation.

    Parameters
    ----------
    s_vectors : SVectors or None
        SVectors object containing the B-matrix and optional second-order
        B-tensors. May be None for pure Cartesian displacements.
    zmat : Zmat
        Molecular coordinate object containing Cartesian coordinates,
        internal-coordinate definitions, masses, and connectivity data.
    eigs : ndarray
        Matrix of normal-mode eigenvectors.
    conv : bool
        If True, enforce angular continuity corrections when converting
        Cartesian coordinates to internal coordinates.
    proj : ndarray
        Projection matrix defining the nonredundant internal coordinate space.
    options : Options
        Runtime options controlling displacement sizes, convergence criteria,
        scaling methods, coordinate systems, and numerical differentiation.
    indices : iterable
        List of coordinate index pairs used when generating second-derivative
        displacements.
    symm_obj : object, optional
        Symmetry object used for symmetry-adapted coordinate generation.
    coord_type : {"internal", "cartesian"}, optional
        Coordinate system used for displacement generation.
        Default is "internal".
    deriv_level : int, optional
        Derivative level to generate:

        * 0 : second derivatives (Hessian displacements)
        * 1 : first derivatives (gradient displacements)

    cma_level : str, optional
        Coordinate mode analysis level used when scaling displacements.
        Default is "B".

    Attributes
    ----------
    B : ndarray
        Wilson B-matrix relating internal and Cartesian coordinates.
    A : ndarray
        First-order transformation matrix from normal coordinates to
        Cartesian displacements.
    eig_inv : ndarray
        Normalized inverse eigenvector matrix.
    n_coord : ndarray
        Reference normal-coordinate values.
    p_disp : ndarray
        Positive displacement geometries.
    m_disp : ndarray
        Negative displacement geometries.
    ref_carts : ndarray
        Reference Cartesian geometry.
    disp : float or ndarray
        Displacement magnitudes used during finite differences.
    proj : ndarray
        Internal-coordinate projection matrix.
    coord_type : str
        Coordinate system used for displacement generation.
    deriv_level : int
        Derivative order being generated.

    Notes
    -----
    For internal-coordinate displacements, Cartesian geometries are generated
    iteratively until the transformed geometry reproduces the target normal
    coordinate displacement within the specified convergence tolerance.

    When ``options.second_order`` is enabled, second-order coordinate
    transformations are included through:

        Δx = A ΔQ + 1/2 A₂ : ΔQΔQ

    where A₂ is constructed from numerical second derivatives of the
    Wilson B-matrix.

    The ordering of internal coordinates must remain consistent with the
    ordering defined throughout the Concordant Modes package, since all
    coordinate transformations and force-constant manipulations depend on
    that convention.
    """
    np.set_printoptions(precision=8, linewidth=240)
    def __init__(
        self,
        s_vectors,
        zmat,
        eigs,
        conv,
        proj,
        options,
        indices,
        symm_obj=None,
        coord_type="internal",
        deriv_level=0,
        cma_level="B",
    ):
        self.options = options
        self.conv = conv
        self.s_vectors = s_vectors
        if self.s_vectors is not None:
            self.B = self.s_vectors.B
        else:
            self.B = []
        self.zmat = zmat
        self.ref_carts = self.zmat.cartesians_a.copy()
        self.ref_carts = np.array(self.ref_carts).astype(float)
        self.u = np.identity(3 * len(zmat.atom_list))
        self.disp = self.options.disp
        self.proj = proj
        self.eigs = eigs
        self.disp_cart = {}
        self.disp_cart["ref"] = self.ref_carts.copy()
        self.indices = indices
        self.symm_obj = symm_obj
        self.deriv_level = deriv_level
        self.coord_type = coord_type
        self.cma_level = cma_level

    def run(self, fc=None):
        np.set_printoptions(precision=8, linewidth=240)

        fc = np.asarray(fc) if fc is not None else np.array([])

        self._build_eig_inv()

        if self.coord_type == "internal":
            self._run_internal(fc)

        elif self.coord_type == "cartesian":
            self._run_cartesian()

        else:
            raise RuntimeError("coord_type must be either 'cartesian' or 'internal'.")

    def _build_eig_inv(self, proj_tol=1.0e-3):
        """
        Invert and normalize the eigenvector matrix.
        """

        self.eig_inv = LA.inv(self.eigs)

        for i, row in enumerate(self.eig_inv):

            row /= LA.norm(row)

            thresh = np.max(np.abs(row)) * proj_tol
            row[np.abs(row) < thresh] = 0.0

            self.eig_inv[i] = row

    def _build_mass_matrix(self):
        """
        Construct inverse mass-weight matrix.
        """

        masses = np.asarray(self.zmat.masses, dtype=float)

        inv_masses = np.where(
            np.asarray(self.zmat.atom_list) == "X",
            0.0,
            1.0 / masses,
        )

        return np.diag(np.repeat(inv_masses, 3))

    # Internal displacement functions here:
    def _run_internal(self, fc):

        u = self._build_mass_matrix()

        self.A = self.compute_A(
            self.B.copy(),
            self.proj,
            self.eig_inv,
            u,
            cma_level=self.cma_level,
        )

        self.n_coord = self.int_c(
            self.ref_carts,
            self.eig_inv,
            self.proj,
        )

        self._print_normal_coordinates()

        self._build_displacements(fc)

        A2 = self._build_second_order_A2()

        if self.deriv_level == 1:
            self._generate_internal_first_deriv(A2)

        elif self.deriv_level == 0:
            self._generate_internal_second_deriv(A2)

        else:
            raise RuntimeError("Only energy and gradient derivatives are supported.")

    def _print_normal_coordinates(self):

        print("Normal Coordinate Values:")

        for i, value in enumerate(self.n_coord, start=1):
            print(f"Normal Coordinate #{i:<4}: {value: 3.5f}")

    def _build_displacements(self, fc):

        self.Disp = self.disp
        self.disp = np.full(len(self.n_coord), self.Disp)

        #
        # Reduced displacements
        #
        if self.options.reduced_disp and len(fc):

            scale = self.options.reduced_disp_size

            self.disp = np.array(
                [scale / abs(fc[i, i]) ** 0.25 for i in range(len(self.disp))]
            )

            print("Reduced displacements")
            print(self.disp)

        #
        # CMA scaling
        #
        elif self.options.scaled_disp and self.cma_level == "B":

            for i in range(len(self.disp)):

                self.disp[i] /= np.max(self.proj.T[i])
                self.disp[i] /= np.max(self.eig_inv[i])
                self.disp[i] *= LA.norm(self.eig_inv[i])

    def _build_second_order_A2(self):

        if not self.options.second_order:
            return None

        L = inv(self.eig_inv)

        B2 = np.einsum(
            "ir,ipq->rpq",
            self.proj,
            self.s_vectors.B2,
        )

        B2 = np.einsum(
            "ri,ipq->rpq",
            L,
            B2,
        )

        return self.compute_A2(B2, self.A)

    def _generate_internal_first_deriv(self, A2):

        n = len(self.eigs)

        p_disp = np.empty(n, dtype=object)
        m_disp = np.empty(n, dtype=object)

        for i in range(n):

            disp = np.zeros(n)
            disp[i] = self.disp[i]

            p_disp[i] = self.coord_convert(
                disp,
                self.n_coord.copy(),
                self.ref_carts.copy(),
                self.A.copy(),
                False,
                self.zmat,
                self.options,
                A2=A2,
            )

            m_disp[i] = self.coord_convert(
                -disp,
                self.n_coord.copy(),
                self.ref_carts.copy(),
                self.A.copy(),
                False,
                self.zmat,
                self.options,
                A2=A2,
            )

        self.p_disp = p_disp
        self.m_disp = m_disp

    def _generate_internal_second_deriv(self, A2):

        n = len(self.eigs)

        p_disp = np.empty((n, n), dtype=object)
        m_disp = np.empty((n, n), dtype=object)

        for i, j in self.indices:

            disp = np.zeros(n)

            disp[i] = self.disp[i]
            disp[j] = self.disp[j]

            p_disp[i, j] = self.coord_convert(
                disp,
                self.n_coord.copy(),
                self.ref_carts.copy(),
                self.A.copy(),
                self.options.tight_disp,
                self.zmat,
                self.options,
                A2=A2,
            )

            m_disp[i, j] = self.coord_convert(
                -disp,
                self.n_coord.copy(),
                self.ref_carts.copy(),
                self.A.copy(),
                self.options.tight_disp,
                self.zmat,
                self.options,
                A2=A2,
            )

        self.p_disp = p_disp
        self.m_disp = m_disp

    # Now we have the cartesian functions.
    def _run_cartesian(self):

        self.disp_mag = self.options.disp
        if self.deriv_level == 1:
            self._generate_cartesian_first_deriv()

        elif self.deriv_level == 0:

            if not self.options.molsym_symmetry:
                self._generate_cartesian_second_deriv()

    def _generate_cartesian_first_deriv(self):

        ref = self.ref_carts.flatten()
        n = len(ref)

        p_disp = np.empty(n, dtype=object)
        m_disp = np.empty(n, dtype=object)

        for i in range(n):

            plus = ref.copy()
            minus = ref.copy()

            plus[i] += self.disp
            minus[i] -= self.disp

            p_disp[i] = plus.reshape(-1, 3)
            m_disp[i] = minus.reshape(-1, 3)

        self.p_disp = p_disp
        self.m_disp = m_disp

    def _generate_cartesian_second_deriv(self):

        ref = self.ref_carts.flatten()
        n = len(ref)

        p_disp = np.empty((n, n), dtype=object)
        m_disp = np.empty((n, n), dtype=object)

        for i, j in self.indices:

            plus = ref.copy()
            minus = ref.copy()

            plus[i] += self.disp
            plus[j] += self.disp

            minus[i] -= self.disp
            minus[j] -= self.disp

            p_disp[i, j] = plus.reshape(-1, 3)
            m_disp[i, j] = minus.reshape(-1, 3)

        self.p_disp = p_disp
        self.m_disp = m_disp

    def int_c(self, carts, eig_inv, proj):
        """
        Compute all internal coordinates from Cartesian coordinates.

        The ordering of generated internal coordinates is significant and must
        remain consistent with the rest of the package.
        """
        carts = np.asarray(carts, dtype=float)
        int_coord = []

        def get_point(indices):
            """
            Return either:
            - a Cartesian coordinate for a single atom index
            - a centroid for a group of atom indices
            """
            if np.ndim(indices) == 0:
                return carts[int(indices) - 1]

            coords = np.asarray(
                [carts[int(i) - 1] for i in indices],
                dtype=float,
            )
            return self.calc_Centroid(coords)

        def get_points(index_set):
            return [get_point(indices) for indices in index_set]

        #
        # Bonds
        #
        for inds in self.zmat.bond_indices:
            x1, x2 = get_points(inds)
            int_coord.append(self.calc_bond(x1, x2))

        #
        # Center-of-mass distances
        #
        for mol1, mol2 in self.zmat.rcom_indices:
            int_coord.append(self.calc_Rcom(mol1, mol2, carts))

        #
        # Angles
        #
        for i, inds in enumerate(self.zmat.angle_indices):
            x1, x2, x3 = get_points(inds)

            angle = self.calc_angle(x1, x2, x3)

            if self.conv:
                ref = float(
                    self.zmat.variable_dictionary_a[self.zmat.angle_variables[i]]
                )

                if ref > 180.0 or ref < 0.0:
                    angle = 2 * np.pi - angle

            int_coord.append(angle)

        #
        # Torsions
        #
        for i, inds in enumerate(self.zmat.torsion_indices):
            x1, x2, x3, x4 = get_points(inds)

            tors = self.calc_tors(x1, x2, x3, x4)

            if self.conv:
                ref = float(
                    self.zmat.variable_dictionary_a[self.zmat.torsion_variables[i]]
                )

                tors_deg = tors * 180.0 / np.pi

                if ref > 135.0 and tors_deg < -135.0:
                    tors += 2 * np.pi
                elif ref < -135.0 and tors_deg > 135.0:
                    tors -= 2 * np.pi

            int_coord.append(tors)

        #
        # Out-of-plane bends
        #
        for i, inds in enumerate(self.zmat.oop_indices):
            x1, x2, x3, x4 = get_points(inds)

            oop = self.calc_OOP(x1, x2, x3, x4)

            if self.conv:
                ref = float(self.zmat.variable_dictionary_a[self.zmat.oop_variables[i]])

                if ref > 180.0:
                    oop -= 2 * np.pi
                elif ref < -180.0:
                    oop += 2 * np.pi

            int_coord.append(oop)

        #
        # Linear bends
        #
        linear_coordinate_sets = [
            (self.zmat.lin_indices, self.calc_Lin),
            (self.zmat.linx_indices, self.calc_Linx),
            (self.zmat.liny_indices, self.calc_Liny),
        ]

        for index_sets, func in linear_coordinate_sets:
            for inds in index_sets:
                x1, x2, x3, x4 = get_points(inds)
                int_coord.append(func(x1, x2, x3, x4))

        int_coord = np.asarray(int_coord)

        return eig_inv @ (proj.T @ int_coord)

    def calc_bond(self, x1, x2):
        r = LA.norm(x1 - x2)
        return r

    def calc_angle(self, x1, x2, x3):
        a = np.dot(x1 - x2, x3 - x2) / (self.calc_bond(x1, x2) * self.calc_bond(x3, x2))
        a = np.arccos(a)
        return a

    def calc_tors(self, x1, x2, x3, x4):
        e1 = (x2 - x1) / self.calc_bond(x1, x2)
        e2 = (x3 - x2) / self.calc_bond(x2, x3)
        e3 = (x4 - x3) / self.calc_bond(x3, x4)
        s = np.dot(-e1, np.cross(-e2, e3))
        c = np.dot(np.cross(-e1, e2), np.cross(-e2, e3))
        t = np.arctan2(s, c)
        return t

    def calc_twist(self, x1, x2, x3, x4, x5):
        e1 = (x1 - x2) / self.calc_bond(x1, x2)
        e3 = (x3 - x2) / self.calc_bond(x2, x3)
        e4 = (x3 - x4) / self.calc_bond(x4, x3)
        e5 = (x3 - x5) / self.calc_bond(x5, x3)
        phi1 = self.calc_angle(x1, x2, x3)
        phi2 = self.calc_angle(x4, x3, x5)
        c = np.dot(np.cross(e1, e3), np.cross(e4, e5)) / (np.sin(phi1) * np.sin(phi2))
        t = np.arccos(c)

        return t

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
        e1 = (x1 - x2) / self.calc_bond(x1, x2)
        e3 = (x3 - x2) / self.calc_bond(x3, x2)
        e4 = (x4 - x2) / self.calc_bond(x4, x2)
        phi = self.calc_angle(x3, x2, x4)
        theta = np.arcsin(np.dot(np.cross(e3, e4) / np.sin(phi), e1))
        return theta

    def calc_Lin(self, x1, x2, x3, x4):
        e1 = (x1 - x2) / self.calc_bond(x1, x2)
        e3 = (x3 - x2) / self.calc_bond(x3, x2)
        e4 = (x4 - x2) / self.calc_bond(x4, x2)
        theta = np.arcsin(np.dot(e4, np.cross(e3, e1)))
        return theta

    def calc_Linx(self, x1, x2, x3, x4):
        e1 = (x2 - x1) / self.calc_bond(x1, x2)
        e2 = (x3 - x2) / self.calc_bond(x2, x3)
        e3 = (x4 - x3) / self.calc_bond(x3, x4)
        theta = self.calc_angle(x1, x2, x3)
        s = np.dot(np.cross(-e1, e2), np.cross(-e2, e3))
        ax = s / np.sin(theta)
        return ax

    def calc_Liny(self, x1, x2, x3, x4):
        e1 = (x2 - x1) / self.calc_bond(x1, x2)
        e2 = (x3 - x2) / self.calc_bond(x2, x3)
        e3 = (x4 - x3) / self.calc_bond(x3, x4)
        theta = self.calc_angle(x1, x2, x3)
        s = np.dot(-e1, np.cross(-e2, e3))
        ay = s / np.sin(theta)
        return ay

    def calc_Rcom(self, mol1, mol2, carts):
        mass = [masses.get_mass(label) for label in self.zmat.atom_list]
        mol1_carts = np.array([[0.0, 0.0, 0.0]])
        mol1_masses = np.array([])
        for j in range(len(mol1)):
            mol1_carts = np.append(mol1_carts, [carts[int(mol1[j]) - 1]], axis=0)
            mol1_masses = np.append(mol1_masses, mass[int(mol1[j]) - 1])
        mol1_carts = mol1_carts[1:]
        mol2_carts = np.array([[0.0, 0.0, 0.0]])
        mol2_masses = np.array([])
        for j in range(len(mol2)):
            mol2_carts = np.append(mol2_carts, [carts[int(mol2[j]) - 1]], axis=0)
            mol2_masses = np.append(mol2_masses, mass[int(mol2[j]) - 1])
        mol2_carts = mol2_carts[1:]

        mass_weighted1 = np.dot(mol1_masses, mol1_carts)
        mass_weighted2 = np.dot(mol2_masses, mol2_carts)
        com1 = mass_weighted1 / np.sum(mol1_masses)
        com2 = mass_weighted2 / np.sum(mol2_masses)

        com_len = com1 - com2

        rc = LA.norm(com_len)
        return rc

    def calc_Centroid(self, carts):
        c = [0.0, 0.0, 0.0]
        L = len(carts)
        for i in carts:
            c += i
        c /= L

        return c

    def coord_convert(
        self,
        n_disp,
        n_coord,
        ref_carts,
        A,
        tight_disp,
        zmat,
        options,
        A2=np.array([]),
    ):
        disp = n_disp.copy()
        new_n = n_coord + n_disp
        new_carts = np.array(ref_carts).astype(float)
        tolerance = options.transf_tol
        for i in range(options.transf_max_it):
            cart_disp = np.dot(n_disp, A)
            np.set_printoptions(precision=6, linewidth=240)
            if self.options.second_order:
                cart_disp += 0.5 * np.einsum("ipq,p,q->i", A2, n_disp, n_disp)
            cart_disp_shaped = np.reshape(cart_disp, (-1, 3))
            new_carts += cart_disp_shaped
            coord_check = self.int_c(new_carts, self.eig_inv, self.proj)
            n_disp = new_n - coord_check

            if tight_disp:
                sVec = s_vec(zmat, options)
                sVec.run(new_carts, False)
                A = self.compute_A(
                    sVec.B, self.proj, self.eig_inv, self.zmat.mass_weight
                )
            if LA.norm(n_disp) < tolerance:
                break
        if LA.norm(n_disp) > tolerance:
            print("This displacement did not converge.")
            print(np.where(abs(disp) == self.Disp))
            print("Norm:")
            print(LA.norm(n_disp))
            print("Tolerance:")
            print(tolerance)
        return new_carts

    def compute_A(self, B, proj, eig_inv, u, cma_level="B"):
        """
        Construct 'A', the commented lines may be useful for getting
        intensities later. The BB^T product must be linearly independent
        to work, so it will be projected into the symmetry adapted basis.
        """

        L = inv(eig_inv)

        A = LA.pinv(B)  # (3N x s)
        A = np.dot(A, proj)  # (3N x S)
        A = A.T  # (S x 3N)

        # This could be necessary
        # for intensities.
        # B = np.dot(proj.T,B)
        # u = np.eye(len(B.T))
        # # A = inv(B.dot(np.sqrt(u)).dot(B.T)) # (s x s)
        # A = inv(B.dot(u).dot(B.T)) # (s x s)
        # A = (B.T).dot(A) # (3N x s)
        # # A = np.sqrt(u).dot(A)
        # # A = u.dot(A)
        # A = A.T  # (S x 3N)

        # This step modifies A to convert from normal coords to carts.
        A = np.dot(L.T, A)  # (Q x 3N)

        return A

    def compute_A2(self, B2, A):
        C2 = np.einsum("rij,pi,qj->rpq", B2, A, A)

        A2 = np.einsum("rpq,ri->ipq", C2, A)

        return A2
