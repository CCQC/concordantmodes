import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from concordantmodes.s_vectors import SVectors as s_vec


class TransfDisp(object):
    """
    The purpose of this script is to perform an iterative transformation
    between internal coordinate displacements and cartesian displacements,
    using a first order, linear B-tensor transformation.

    It is necessary to construct an inverse B-tensor called 'A', where,

    A = uB^T(BuB^T)^-1.

    u can simply be the I_3N identity matrix, so long as frequency intensities
    are not desired.

    If this is the case, then A simply becomes the following:

    A = B^T(BB^T)^-1.

    Where in fact, BB^T = G (non-mass-weighted)

    """

    def __init__(
        self,
        s_vectors,
        zmat,
        disp,
        eigs,
        conv,
        disp_tol,
        ted,
        options,
        indices,
        GF=None,
        deriv_level=0,
    ):
        # As is generally the case for Programmers, GF = None by default.
        self.disp_tol = disp_tol
        self.conv = conv
        self.s_vectors = s_vectors
        self.zmat = zmat
        self.ref_carts = zmat.cartesians_final.copy()
        self.ref_carts = np.array(self.ref_carts).astype(float)
        self.u = np.identity(3 * len(zmat.atom_list))
        self.disp = disp
        self.ted = ted
        self.eigs = eigs
        self.options = options
        self.disp_cart = {}
        self.disp_cart["ref"] = self.ref_carts.copy()
        self.indices = indices
        if GF:
            self.GF = GF
        self.deriv_level = deriv_level

    def run(self):

        self.B = self.s_vectors.B.copy()  # (redundant internals (s) x cartesians (3N))
        # Invert the L-matrix and then normalize the rows.
        proj_tol = 1.0e-3
        self.eig_inv = LA.inv(self.eigs)  # (Normal modes (Q) x Sym internals (S) )
        for i in range(len(self.eig_inv)):
            self.eig_inv[i] = self.eig_inv[i] / LA.norm(self.eig_inv[i])
            self.eig_inv[i][
                np.abs(self.eig_inv[i]) < np.max(np.abs(self.eig_inv[i])) * proj_tol
            ] = 0

        # Compute the A-matrix to convert from normal coordinates to cartesians
        self.A = self.compute_A(
            self.s_vectors.B.copy(), self.ted.proj, self.eig_inv, self.zmat.mass_weight
        )

        # Generate the normal coordinate values for the reference structure
        self.n_coord = self.int_c(self.ref_carts, self.eig_inv, self.ted.proj)
        # self.disp_sym = np.zeros(len(self.n_coord))

        print("Normal Coordinate Values:")
        for i in range(len(self.n_coord)):
            print("Normal Coordinate #{:<4n}: {: 3.5f}".format(i + 1, self.n_coord[i]))
            # if abs(self.n_coord[i]) < self.disp_tol:
            # self.disp_sym[i] = 1

        # print("List of symmetric displacements according to disp_tol:")
        # print(self.disp_sym)

        # Next, we will have to specify our Normal mode internal coordinate
        # displacement sizes

        Disp = self.disp
        self.disp = []
        for i in range(len(self.n_coord)):
            self.disp.append(Disp)

        # This code will be useful for submitting the computation with
        # reduced displacements
        # if self.options.reducedDisp:
        # self.Freq = inv(np.diag(self.GF.Freq.copy()))
        # redDisp = np.dot(self.disp,self.Freq)
        # redDisp = redDisp / min(redDisp)
        # redDisp = redDisp*self.disp
        # self.disp = redDisp

        # raise RuntimeError

        # This code loops through a list of indices that the force constants will be computed at and
        # generates the displacements for the diagonal and (if specified) the off-diagonals. Where
        # the displacement matrix D[i,j] = D[j,i].
        # a = p_disp.shape[0]
        disp = np.zeros(len(self.eigs.T))
        buff = disp.copy()
        if not self.deriv_level:
            p_disp = np.zeros((len(self.eigs), len(self.eigs)), dtype=object)
            m_disp = np.zeros((len(self.eigs), len(self.eigs)), dtype=object)
            for index in self.indices:
                i, j = index[0], index[1]
                disp[i] = self.disp[i]
                disp[j] = self.disp[j]
                # a = self.coord_convert(
                # print('['+str(i)+', '+str(j)+']')
                p_disp[i, j] = self.coord_convert(
                    disp,
                    self.n_coord.copy(),
                    self.ref_carts.copy(),
                    50,
                    1.0e-7,
                    self.A.copy(),
                    False,
                    self.zmat,
                    self.options,
                )
                # print(a.shape)
                # raise RuntimeError
                m_disp[i, j] = self.coord_convert(
                    -disp,
                    self.n_coord.copy(),
                    self.ref_carts.copy(),
                    50,
                    1.0e-7,
                    self.A.copy(),
                    False,
                    self.zmat,
                    self.options,
                )
                disp = buff.copy()

            # raise RuntimeError
        elif self.deriv_level == 1:
            p_disp = np.zeros(len(self.eigs), dtype=object)
            m_disp = np.zeros(len(self.eigs), dtype=object)
            for i in range(len(self.eigs)):
                disp[i] = self.disp[i]
                # print(disp)
                p_disp[i] = self.coord_convert(
                    disp,
                    self.n_coord.copy(),
                    self.ref_carts.copy(),
                    50,
                    1.0e-9,
                    self.A.copy(),
                    False,
                    self.zmat,
                    self.options,
                )
                m_disp[i] = self.coord_convert(
                    -disp,
                    self.n_coord.copy(),
                    self.ref_carts.copy(),
                    50,
                    1.0e-9,
                    self.A.copy(),
                    False,
                    self.zmat,
                    self.options,
                )
                disp = buff.copy()
            # print(p_disp)
            # print(m_disp)
            # raise RuntimeError
        else:
            print(
                "Only energy and gradient derivatives are supported. Check your deriv_level_init keyword."
            )
            raise RuntimeError
        self.p_disp = p_disp
        self.m_disp = m_disp

    def int_c(self, carts, eig_inv, proj):
        # This is a function that computes all currently implemented and
        # specified internal coordinates from the desired cartesian
        # coordinates. The internal coordinate values will be appended
        # together in the order of the for-loops below. All other methods
        # in this package which use the int_c generated variables MUST
        # abide by this ordering.

        # tol = 1.0e-3
        int_coord = np.array([])
        for i in range(len(self.zmat.bond_indices)):
            x1 = np.array(carts[int(self.zmat.bond_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.bond_indices[i][1]) - 1]).astype(float)
            int_coord = np.append(int_coord, self.calc_bond(x1, x2))
        for i in range(len(self.zmat.angle_indices)):
            x1 = np.array(carts[int(self.zmat.angle_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.angle_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.angle_indices[i][2]) - 1]).astype(float)
            a = self.calc_angle(x1, x2, x3)
            if self.conv:
                condition_1 = (
                    float(
                        self.zmat.variable_dictionary_final[
                            self.zmat.angle_variables[i]
                        ]
                    )
                    > 180.0
                )
                condition_2 = (
                    float(
                        self.zmat.variable_dictionary_final[
                            self.zmat.angle_variables[i]
                        ]
                    )
                    < 0.0
                )
                if condition_1 or condition_2:
                    a = 2 * np.pi - a
            int_coord = np.append(int_coord, a)
        for i in range(len(self.zmat.torsion_indices)):
            x1 = np.array(carts[int(self.zmat.torsion_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.torsion_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.torsion_indices[i][2]) - 1]).astype(float)
            x4 = np.array(carts[int(self.zmat.torsion_indices[i][3]) - 1]).astype(float)
            t = self.calc_tors(x1, x2, x3, x4)
            if self.conv:
                condition_1 = float(
                    self.zmat.variable_dictionary_final[self.zmat.torsion_variables[i]]
                ) > 135.0 and (t * 180.0 / np.pi < -135.0)
                condition_2 = float(
                    self.zmat.variable_dictionary_final[self.zmat.torsion_variables[i]]
                ) < -135.0 and (t * 180.0 / np.pi > 135.0)
                if condition_1:
                    t += 2 * np.pi
                if condition_2:
                    t -= 2 * np.pi
            int_coord = np.append(int_coord, t)
        for i in range(len(self.zmat.oop_indices)):
            x1 = np.array(carts[int(self.zmat.oop_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.oop_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.oop_indices[i][2]) - 1]).astype(float)
            x4 = np.array(carts[int(self.zmat.oop_indices[i][3]) - 1]).astype(float)
            o = self.calc_OOP(x1, x2, x3, x4)
            if self.conv:
                condition_1 = (
                    float(
                        self.zmat.variable_dictionary_final[self.zmat.oop_variables[i]]
                    )
                    > 180.0
                )
                condition_2 = (
                    float(
                        self.zmat.variable_dictionary_final[self.zmat.oop_variables[i]]
                    )
                    < -180.0
                )
                if condition_1:
                    o = o - 2 * np.pi
                if condition_2:
                    o = o + 2 * np.pi
            int_coord = np.append(int_coord, o)

        # These angles will have to be tempered at some point in the same manner as above.
        for i in range(len(self.zmat.lin_indices)):
            x1 = np.array(carts[int(self.zmat.lin_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.lin_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.lin_indices[i][2]) - 1]).astype(float)
            x4 = np.array(carts[int(self.zmat.lin_indices[i][3]) - 1]).astype(float)
            l = self.calc_Lin(x1, x2, x3, x4)
            int_coord = np.append(int_coord, l)

        for i in range(len(self.zmat.linx_indices)):
            x1 = np.array(carts[int(self.zmat.linx_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.linx_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.linx_indices[i][2]) - 1]).astype(float)
            x4 = np.array(carts[int(self.zmat.linx_indices[i][3]) - 1]).astype(float)
            lx = self.calc_Linx(x1, x2, x3, x4)
            int_coord = np.append(int_coord, lx)

        for i in range(len(self.zmat.liny_indices)):
            x1 = np.array(carts[int(self.zmat.liny_indices[i][0]) - 1]).astype(float)
            x2 = np.array(carts[int(self.zmat.liny_indices[i][1]) - 1]).astype(float)
            x3 = np.array(carts[int(self.zmat.liny_indices[i][2]) - 1]).astype(float)
            x4 = np.array(carts[int(self.zmat.liny_indices[i][3]) - 1]).astype(float)
            ly = self.calc_Liny(x1, x2, x3, x4)
            int_coord = np.append(int_coord, ly)

        int_coord = np.dot(proj.T, int_coord)
        # int_coord[np.abs(int_coord) < tol*np.max(np.abs(int_coord))] = 0
        int_coord = np.dot(eig_inv, int_coord)
        return int_coord

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

    def coord_convert(
        self,
        n_disp,
        n_coord,
        ref_carts,
        max_iter,
        tolerance,
        A,
        tight_disp,
        zmat,
        options,
    ):
        tol = 1.0e-3
        # conv_iter = 10
        new_n = n_coord + n_disp
        new_carts = np.array(ref_carts).astype(float)
        for i in range(max_iter):
            cart_disp = np.dot(n_disp, A)
            cart_disp_shaped = np.reshape(cart_disp, (-1, 3))
            new_carts += cart_disp_shaped
            coord_check = self.int_c(new_carts, self.eig_inv, self.ted.proj)
            n_disp = new_n - coord_check
            n_disp[np.abs(n_disp) < tol * np.max(np.abs(n_disp))] = 0

            if tight_disp:
                sVec = s_vec(zmat, options, zmat.variable_dictionary_final)
                sVec.run(new_carts, False)
                A = self.compute_A(
                    sVec.B, self.ted.proj, self.eig_inv, self.zmat.mass_weight
                )
            if LA.norm(n_disp) < tolerance:
                break
        if LA.norm(n_disp) > tolerance:
            print("This displacement did not converge.")
            print("Norm:")
            print(LA.norm(n_disp))
            print("Tolerance:")
            print(tolerance)
        return new_carts

    def compute_A(self, B, proj, eig_inv, u):
        """
        Construct 'A', the commented lines may be useful for getting
        intensities later. The BB^T product must be linearly independent
        to work, so it will be projected into the symmetry adapted basis.
        """
        # BT = B.T # (3N x s)
        # A = u.dot(BT)
        # A = np.dot(B,BT) # (s x s)
        # A = LA.pinv(A) # (s x s)
        # A = BT.dot(A) # (3N x s)
        # A = np.dot(A,proj).T # (S x 3N)

        L = inv(eig_inv)
        A = LA.pinv(B)
        A = np.dot(A, proj)  # (3N x S)
        A = A.T  # (S x 3N)

        # This could be necessary
        # for intensities.
        # A = A.dot(np.sqrt(inv(u)))
        # This step modifies A to convert from normal coords to carts.
        A = np.dot(L.T, A)  # (Q x 3N)

        return A
