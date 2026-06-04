import numpy as np
from concordantmodes.ted import TED
from concordantmodes.int2cart import Int2Cart
from numpy import linalg as LA


class SVectors:
    """
    s-vectors: s_a^k = (B_ax^k,B_ay^k,B_az^k)
    Where a refers to the atom #, and k refers to the internal coordinate.
    So, all B-tensors can be contained within the s-vector
    for each atom in the molecular system of interest.
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
            phi_1 = self.compute_phi(-e_1, e_2)
            phi_2 = self.compute_phi(-e_2, e_3)

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
            ay = self.calc_alpha_y(e_1, e_2, e_3)
            phi_1 = self.compute_phi(-e_1, e_2)
            phi_2 = self.compute_phi(-e_2, e_3)

            ly1 = self.compute_LINY1(e_1, e_2, e_3, r_1, phi_1, ay)
            ly2 = self.compute_LINY2(e_1, e_2, e_3, r_1, r_2, phi_1, ay)
            ly4 = self.compute_LINY4(e_1, e_2, e_3, r_2, r_3, phi_1, ay)
            ly3 = -ly1 - ly2 - ly4

            values = [ly1, ly2, ly3, ly4]

            key = f"Ly{i+1}"
            self._build_s_vector(key, groups, values, self.s_4center_dict)

        # The last step will be to concatenate all of the s-vectors into a single B-tensor.
        rows = []

        rows += [v.flatten() for v in self.s_2center_dict.values()]
        rows += [v.flatten() for v in self.s_3center_dict.values()]
        rows += [v.flatten() for v in self.s_4center_dict.values()]

        self.B = np.array(rows)

        tol = 1e-4
        # Now we acquire a linearly independant set of internal coordinates from the diagonalized
        # BB^T Matrix
        if not self.options.man_proj:
            if self.options.coords.upper() != "ZMAT":
                proj, eigs, _ = LA.svd(self.B)
                proj[np.abs(proj) < tol] = 0
                print("proj singular values:")
                print(eigs)
                if B_proj:
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
            for i in range(len(self.B2)):
                for j in range(len(self.B2[0]) - 1):
                    for k in range(j):
                        self.B2[i, j + 1, k] = (
                            self.B2[i, j + 1, k] + self.B2[i, k, j + 1]
                        ) / 2
                        self.B2[i, k, j + 1] = self.B2[i, j + 1, k]
            self.B2 = self.B2.astype(float)
            self.proj = Proj
            self.B = B

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
        a = -ax * e_1 / (r * np.sin(phi_1) ** 2)
        b = -e_2 * (ax * (np.tan(phi_1) ** -1) + np.cos(phi_2)) / (r * np.sin(phi_1))
        c = -e_3 / (r * np.sin(phi_1))
        s = a + b + c
        return s

    def compute_LINX2(self, e_1, e_2, e_3, r_1, r_2, phi_1, phi_2, ax):
        a = (
            e_1
            * (ax * (r_2 - r_1 * np.cos(phi_1)) - r_1 * np.sin(phi_1) * np.cos(phi_2))
            / (r_1 * r_2 * np.sin(phi_1) ** 2)
        )
        b = (
            -e_2
            * (
                ax * (np.tan(phi_1) ** -1) * (r_1 * np.cos(phi_1) - r_2)
                + np.cos(phi_2) * (2 * r_1 * np.cos(phi_1) - r_2)
            )
            / (r_1 * r_2 * np.sin(phi_1))
        )
        c = -e_3 * (r_1 * np.cos(phi_1) - r_2) / (r_1 * r_2 * np.sin(phi_1))
        s = a + b + c
        return s

    def compute_LINX4(self, e_1, e_2, e_3, r, phi, ax):
        a = e_1 / (r * np.sin(phi))
        b = e_2 * (np.tan(phi) ** -1) / r
        c = ax * e_3 / r
        s = a + b + c
        return s

    def compute_LINY1(self, e_1, e_2, e_3, r, phi, ay):
        a = -ay * (np.tan(phi) ** -1) * (e_1 * np.cos(phi) + e_2) / (r * np.sin(phi))
        b = np.cross(e_2, -e_3) / (r * np.sin(phi))
        c = -e_1 * ay / r
        s = a + b + c
        return s

    def compute_LINY2(self, e_1, e_2, e_3, r_1, r_2, phi, ay):
        a = (
            -np.cos(phi)
            * ay
            * (e_1 * (r_1 - r_2 * np.cos(phi)) - e_2 * (r_2 - r_1 * np.cos(phi)))
            / (r_1 * r_2 * np.sin(phi) ** 2)
        )
        b = (r_2 * np.cross(e_2, -e_3) + r_1 * np.cross(-e_3, e_1)) / (
            r_1 * r_2 * np.sin(phi)
        )
        c = ay * (e_1 * r_2 + e_2 * r_1) / (r_1 * r_2)
        s = a + b + c
        return s

    def compute_LINY4(self, e_1, e_2, e_3, r_1, r_2, phi, ay):
        s = ((np.sin(phi) ** -1) * np.cross(e_1, e_2) + e_3 * ay) / r_2
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
        theta = self.compute_phi(e_1, e_2)
        s = np.dot(np.cross(e_1, e_2), np.cross(-e_2, e_3))
        ax = s / np.sin(theta)
        return ax

    def calc_alpha_y(self, e_1, e_2, e_3):
        theta = self.compute_phi(e_1, e_2)
        s = np.dot(e_1, np.cross(-e_2, e_3))
        ay = s / np.sin(theta)
        return ay

    def calc_r_com(self, cart1, cart2, mass1, mass2):
        rc1 = self.int2cart.COM(cart1, mass1)
        rc2 = self.int2cart.COM(cart2, mass2)
        print(rc1)
        print(rc2)
        rc = rc1 - rc2
        rc = LA.norm(rc)
        print(rc)
        return rc

    def num_differentiate(self, B_list_p, B_list_m):
        disp_size = self.options.disp

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
        self.run(B_disp.p_disp[0], False, proj=self.proj, second_order=False)
        B_list_p = np.array([self.B.copy()], dtype=object)
        self.run(B_disp.m_disp[0], False, proj=self.proj, second_order=False)
        B_list_m = np.array([self.B.copy()], dtype=object)

        for i in range(len(B_disp.p_disp) - 1):
            self.run(B_disp.p_disp[i + 1], False, second_order=False)
            B_list_p = np.append(B_list_p, [self.B.copy()], axis=0)
            self.run(B_disp.m_disp[i + 1], False, second_order=False)
            B_list_m = np.append(B_list_m, [self.B.copy()], axis=0)

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
