import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from . import masses
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
        coord_type="internal",
        deriv_level=0,
        cubic_indices=np.array([]),
        quartic_indices=np.array([]),
        anharm=False,
    ):
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
        self.deriv_level = deriv_level
        self.coord_type = coord_type
        self.cubic_indices = cubic_indices
        self.quartic_indices = quartic_indices
        self.anharm = anharm

    def run(self, fc=np.array([])):
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

        print("Normal Coordinate Values:")
        for i in range(len(self.n_coord)):
            print("Normal Coordinate #{:<4n}: {: 3.5f}".format(i + 1, self.n_coord[i]))

        # Next, we will have to specify our Normal mode internal coordinate
        # displacement sizes

        if self.coord_type == "internal":
            self.Disp = self.disp
            self.disp = []
            for i in range(len(self.n_coord)):
                self.disp.append(self.Disp)

            # This code will be useful for submitting the computation with
            # reduced displacements
            if self.options.reduced_disp and len(fc):
                disp_size = self.options.reduced_disp_size
                for i in range(len(self.disp)):
                    self.disp[i] = disp_size / np.abs(fc[i, i]) ** 0.25
                print("Reduced displacements")
                print(self.disp)
                buff_disp = np.zeros(len(self.disp))
                for i in range(len(self.disp)):
                    buff_disp[i] = self.disp[i]
                    simple_disp = np.dot(buff_disp, self.eig_inv)
                    simple_disp = np.dot(simple_disp, self.ted.proj.T)
                    print("Reduced displacement in simple internal coordinates:")
                    print(self.disp.copy()[i])
                    print(simple_disp)
                    buff_disp *= 0

            # This code loops through a list of indices that the force constants will be computed at and
            # generates the displacements for the diagonal and (if specified) the off-diagonals. Where
            # the displacement matrix D[i,j] = D[j,i].
            disp = np.zeros(len(self.eigs.T))
            buff = disp.copy()
            if not self.deriv_level:
                if not self.anharm:
                    p_disp = np.zeros((len(self.eigs), len(self.eigs)), dtype=object)
                    m_disp = np.zeros((len(self.eigs), len(self.eigs)), dtype=object)
                    for index in self.indices:
                        i, j = index[0], index[1]
                        disp[i] = self.disp[i]
                        disp[j] = self.disp[j]
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
                    self.p_disp = p_disp
                    self.m_disp = m_disp
                else:
                    if len(self.cubic_indices):
                        self.p_disp_xxx = np.zeros(len(self.eigs), dtype=object)
                        self.m_disp_xxx = np.zeros(len(self.eigs), dtype=object)
                        if len(self.cubic_indices[0]):
                            cubic_disp_xxx_p = []
                            cubic_disp_xxx_m = []
                            i = 0
                            for index in self.cubic_indices[0]:
                                disp[index[0]] = 2 * self.disp[index[0]]
                                cubic_disp_xxx_p.append(disp)
                                cubic_disp_xxx_m.append(-disp)
                                self.p_disp_xxx[i] = self.coord_convert(
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
                                self.m_disp_xxx[i] = self.coord_convert(
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
                                i += 1
                        self.p_disp_xxy = np.zeros(
                            len(self.cubic_indices[1]), dtype=object
                        )
                        self.m_disp_xxy = np.zeros(
                            len(self.cubic_indices[1]), dtype=object
                        )
                        if len(self.cubic_indices[1]):
                            cubic_disp_xxy_p = []
                            cubic_disp_xxy_m = []
                            i = 0
                            for index in self.cubic_indices[1]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                cubic_disp_xxy_p.append(disp)
                                cubic_disp_xxy_m.append(-disp)
                                self.p_disp_xxy[i] = self.coord_convert(
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
                                self.m_disp_xxy[i] = self.coord_convert(
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
                                i += 1
                        self.p_disp_xyz = np.zeros(
                            len(self.cubic_indices[2]), dtype=object
                        )
                        self.m_disp_xyz = np.zeros(
                            len(self.cubic_indices[2]), dtype=object
                        )
                        if len(self.cubic_indices[2]):
                            cubic_disp_xyz_p = []
                            cubic_disp_xyz_m = []
                            i = 0
                            for index in self.cubic_indices[2]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                cubic_disp_xyz_p.append(disp)
                                cubic_disp_xyz_m.append(-disp)
                                self.p_disp_xyz[i] = self.coord_convert(
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
                                self.m_disp_xyz[i] = self.coord_convert(
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
                                i += 1
                    if len(self.quartic_indices):
                        self.p_disp_xxxy = np.zeros(
                            len(self.quartic_indices[1]), dtype=object
                        )
                        self.m_disp_xxxy = np.zeros(
                            len(self.quartic_indices[1]), dtype=object
                        )
                        if len(self.quartic_indices[1]):
                            quartic_disp_xxxy_p = []
                            quartic_disp_xxxy_m = []
                            i = 0
                            for index in self.quartic_indices[1]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                disp[index[3]] += self.disp[index[3]]
                                quartic_disp_xxxy_p.append(disp)
                                quartic_disp_xxxy_m.append(-disp)
                                self.p_disp_xxxy[i] = self.coord_convert(
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
                                self.m_disp_xxxy[i] = self.coord_convert(
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
                                i += 1
                        self.p_disp_xxyy = np.zeros(
                            len(self.quartic_indices[2]), dtype=object
                        )
                        self.m_disp_xxyy = np.zeros(
                            len(self.quartic_indices[2]), dtype=object
                        )
                        if len(self.quartic_indices[2]):
                            quartic_disp_xxyy_p = []
                            quartic_disp_xxyy_m = []
                            i = 0
                            for index in self.quartic_indices[2]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                disp[index[3]] += self.disp[index[3]]
                                quartic_disp_xxyy_p.append(disp)
                                quartic_disp_xxyy_m.append(-disp)
                                self.p_disp_xxyy[i] = self.coord_convert(
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
                                self.m_disp_xxyy[i] = self.coord_convert(
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
                                i += 1
                        self.p_disp_xxyz = np.zeros(
                            len(self.quartic_indices[3]), dtype=object
                        )
                        self.m_disp_xxyz = np.zeros(
                            len(self.quartic_indices[3]), dtype=object
                        )
                        if len(self.quartic_indices[3]):
                            quartic_disp_xxyz_p = []
                            quartic_disp_xxyz_m = []
                            i = 0
                            for index in self.quartic_indices[3]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                disp[index[3]] += self.disp[index[3]]
                                quartic_disp_xxyz_p.append(disp)
                                quartic_disp_xxyz_m.append(-disp)
                                self.p_disp_xxyz[i] = self.coord_convert(
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
                                self.m_disp_xxyz[i] = self.coord_convert(
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
                                i += 1
                        self.p_disp_wxyz = np.zeros(
                            len(self.quartic_indices[4]), dtype=object
                        )
                        self.m_disp_wxyz = np.zeros(
                            len(self.quartic_indices[4]), dtype=object
                        )
                        if len(self.quartic_indices[4]):
                            quartic_disp_wxyz_p = []
                            quartic_disp_wxyz_m = []
                            i = 0
                            for index in self.quartic_indices[4]:
                                disp[index[0]] += self.disp[index[0]]
                                disp[index[1]] += self.disp[index[1]]
                                disp[index[2]] += self.disp[index[2]]
                                disp[index[3]] += self.disp[index[3]]
                                quartic_disp_wxyz_p.append(disp)
                                quartic_disp_wxyz_m.append(-disp)
                                self.p_disp_wxyz[i] = self.coord_convert(
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
                                self.m_disp_wxyz[i] = self.coord_convert(
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
                                i += 1

            elif self.deriv_level == 1:
                p_disp = np.zeros(len(self.eigs), dtype=object)
                m_disp = np.zeros(len(self.eigs), dtype=object)
                for i in range(len(self.eigs)):
                    disp[i] = self.disp[i]
                    p_disp[i] = self.coord_convert(
                        disp,
                        self.n_coord.copy(),
                        self.ref_carts.copy(),
                        50,
                        1.0e-10,
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
                        1.0e-10,
                        self.A.copy(),
                        False,
                        self.zmat,
                        self.options,
                    )
                    disp = buff.copy()
                self.p_disp = p_disp
                self.m_disp = m_disp
            else:
                print(
                    "Only energy and gradient derivatives are supported. Check your deriv_level_init keyword."
                )
                raise RuntimeError
        elif self.coord_type == "cartesian":
            if self.deriv_level == 1:
                p_disp = np.zeros(len(self.ref_carts.flatten()), dtype=object)
                m_disp = np.zeros(len(self.ref_carts.flatten()), dtype=object)
                # print("THIS IS THE DISP SIZE")
                # print(self.disp)
                for i in range(len(self.ref_carts.flatten())):
                    p_disp[i] = self.ref_carts.flatten().copy()
                    m_disp[i] = self.ref_carts.flatten().copy()
                    p_disp[i][i] = p_disp[i][i] + self.disp
                    m_disp[i][i] = m_disp[i][i] - self.disp
                    p_disp[i] = np.reshape(p_disp[i], (-1, 3))
                    m_disp[i] = np.reshape(m_disp[i], (-1, 3))

                self.p_disp = p_disp
                self.m_disp = m_disp
            elif not self.deriv_level:
                p_disp = np.zeros((len(self.ref_carts.flatten()),len(self.ref_carts.flatten())), dtype=object)
                m_disp = np.zeros((len(self.ref_carts.flatten()),len(self.ref_carts.flatten())), dtype=object)
                # print("THIS IS THE DISP SIZE")
                # print(self.disp)
                for index in self.indices:
                    i, j = index[0], index[1]
                    if i == j:
                        p_disp[i,i] = self.ref_carts.flatten().copy()
                        m_disp[i,i] = self.ref_carts.flatten().copy()
                        p_disp[i,i][i] += self.disp
                        m_disp[i,i][i] -= self.disp
                        p_disp[i,i] = np.reshape(p_disp[i,i], (-1, 3))
                        m_disp[i,i] = np.reshape(m_disp[i,i], (-1, 3))
                    else:
                        p_disp[i,j] = self.ref_carts.flatten().copy()
                        m_disp[i,j] = self.ref_carts.flatten().copy()
                        p_disp[i,j][i] += self.disp
                        p_disp[i,j][j] += self.disp
                        m_disp[i,j][i] -= self.disp
                        m_disp[i,j][j] -= self.disp
                        p_disp[i,j] = np.reshape(p_disp[i,j], (-1, 3))
                        m_disp[i,j] = np.reshape(m_disp[i,j], (-1, 3))

                self.p_disp = p_disp
                self.m_disp = m_disp

        else:
            print(
                "Please input a displacement coordinate type of either cartesian or internal."
            )
            raise RuntimeError

    def int_c(self, carts, eig_inv, proj):
        # This is a function that computes all currently implemented and
        # specified internal coordinates from the desired cartesian
        # coordinates. The internal coordinate values will be appended
        # together in the order of the for-loops below. All other methods
        # in this package which use the int_c generated variables MUST
        # abide by this ordering.
        int_coord = np.array([])
        for i in range(len(self.zmat.bond_indices)):
            for j in self.zmat.bond_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.bond_indices[i]
                    bond_carts = []
                    for k in indies[0]:
                        bond_carts = np.append(
                            bond_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    bond_carts = np.reshape(bond_carts, (-1, 3))
                    x1 = self.calc_Centroid(bond_carts)

                    bond_carts = []
                    for k in indies[1]:
                        bond_carts = np.append(
                            bond_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    bond_carts = np.reshape(bond_carts, (-1, 3))
                    x2 = self.calc_Centroid(bond_carts)
                else:
                    x1 = np.array(carts[int(self.zmat.bond_indices[i][0]) - 1]).astype(
                        float
                    )
                    x2 = np.array(carts[int(self.zmat.bond_indices[i][1]) - 1]).astype(
                        float
                    )

            int_coord = np.append(int_coord, self.calc_bond(x1, x2))

        for i in range(len(self.zmat.rcom_indices)):
            mol1 = self.zmat.rcom_indices[i][0]
            mol2 = self.zmat.rcom_indices[i][1]
            rc = self.calc_Rcom(mol1, mol2, carts)
            int_coord = np.append(int_coord, rc)

        for i in range(len(self.zmat.angle_indices)):
            for j in self.zmat.angle_indices[i]:
                Len = len(np.array(j).shape)
                indies = self.zmat.angle_indices[i]
                if Len:
                    angle_carts = []
                    for k in indies[0]:
                        angle_carts = np.append(
                            angle_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    angle_carts = np.reshape(angle_carts, (-1, 3))
                    x1 = self.calc_Centroid(angle_carts)

                    angle_carts = []
                    for k in indies[1]:
                        angle_carts = np.append(
                            angle_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    angle_carts = np.reshape(angle_carts, (-1, 3))
                    x2 = self.calc_Centroid(angle_carts)

                    angle_carts = []
                    for k in indies[2]:
                        angle_carts = np.append(
                            angle_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    angle_carts = np.reshape(angle_carts, (-1, 3))
                    x3 = self.calc_Centroid(angle_carts)
                else:
                    x1 = np.array(carts[int(indies[0]) - 1]).astype(float)
                    x2 = np.array(carts[int(indies[1]) - 1]).astype(float)
                    x3 = np.array(carts[int(indies[2]) - 1]).astype(float)

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
            for j in self.zmat.torsion_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.torsion_indices[i]
                    torsion_carts = []
                    for k in indies[0]:
                        torsion_carts = np.append(
                            torsion_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    torsion_carts = np.reshape(torsion_carts, (-1, 3))
                    x1 = self.calc_Centroid(torsion_carts)

                    torsion_carts = []
                    for k in indies[1]:
                        torsion_carts = np.append(
                            torsion_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    torsion_carts = np.reshape(torsion_carts, (-1, 3))
                    x2 = self.calc_Centroid(torsion_carts)

                    torsion_carts = []
                    for k in indies[2]:
                        torsion_carts = np.append(
                            torsion_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    torsion_carts = np.reshape(torsion_carts, (-1, 3))
                    x3 = self.calc_Centroid(torsion_carts)

                    torsion_carts = []
                    for k in indies[3]:
                        torsion_carts = np.append(
                            torsion_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    torsion_carts = np.reshape(torsion_carts, (-1, 3))
                    x4 = self.calc_Centroid(torsion_carts)
                else:
                    x1 = np.array(
                        carts[int(self.zmat.torsion_indices[i][0]) - 1]
                    ).astype(float)
                    x2 = np.array(
                        carts[int(self.zmat.torsion_indices[i][1]) - 1]
                    ).astype(float)
                    x3 = np.array(
                        carts[int(self.zmat.torsion_indices[i][2]) - 1]
                    ).astype(float)
                    x4 = np.array(
                        carts[int(self.zmat.torsion_indices[i][3]) - 1]
                    ).astype(float)

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
            for j in self.zmat.oop_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.oop_indices[i]
                    oop_carts = []
                    for k in indies[0]:
                        oop_carts = np.append(
                            oop_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    oop_carts = np.reshape(oop_carts, (-1, 3))
                    x1 = self.calc_Centroid(oop_carts)

                    oop_carts = []
                    for k in indies[1]:
                        oop_carts = np.append(
                            oop_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    oop_carts = np.reshape(oop_carts, (-1, 3))
                    x2 = self.calc_Centroid(oop_carts)

                    oop_carts = []
                    for k in indies[2]:
                        oop_carts = np.append(
                            oop_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    oop_carts = np.reshape(oop_carts, (-1, 3))
                    x3 = self.calc_Centroid(oop_carts)

                    oop_carts = []
                    for k in indies[3]:
                        oop_carts = np.append(
                            oop_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    oop_carts = np.reshape(oop_carts, (-1, 3))
                    x4 = self.calc_Centroid(oop_carts)
                else:
                    x1 = np.array(carts[int(self.zmat.oop_indices[i][0]) - 1]).astype(
                        float
                    )
                    x2 = np.array(carts[int(self.zmat.oop_indices[i][1]) - 1]).astype(
                        float
                    )
                    x3 = np.array(carts[int(self.zmat.oop_indices[i][2]) - 1]).astype(
                        float
                    )
                    x4 = np.array(carts[int(self.zmat.oop_indices[i][3]) - 1]).astype(
                        float
                    )

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
            for j in self.zmat.lin_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.lin_indices[i]
                    lin_carts = []
                    for k in indies[0]:
                        lin_carts = np.append(
                            lin_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    lin_carts = np.reshape(lin_carts, (-1, 3))
                    x1 = self.calc_Centroid(lin_carts)

                    lin_carts = []
                    for k in indies[1]:
                        lin_carts = np.append(
                            lin_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    lin_carts = np.reshape(lin_carts, (-1, 3))
                    x2 = self.calc_Centroid(lin_carts)

                    lin_carts = []
                    for k in indies[2]:
                        lin_carts = np.append(
                            lin_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    lin_carts = np.reshape(lin_carts, (-1, 3))
                    x3 = self.calc_Centroid(lin_carts)

                    lin_carts = []
                    for k in indies[3]:
                        lin_carts = np.append(
                            lin_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    lin_carts = np.reshape(lin_carts, (-1, 3))
                    x4 = self.calc_Centroid(lin_carts)
                else:
                    x1 = np.array(carts[int(self.zmat.lin_indices[i][0]) - 1]).astype(
                        float
                    )
                    x2 = np.array(carts[int(self.zmat.lin_indices[i][1]) - 1]).astype(
                        float
                    )
                    x3 = np.array(carts[int(self.zmat.lin_indices[i][2]) - 1]).astype(
                        float
                    )
                    x4 = np.array(carts[int(self.zmat.lin_indices[i][3]) - 1]).astype(
                        float
                    )
            l = self.calc_Lin(x1, x2, x3, x4)
            int_coord = np.append(int_coord, l)

        for i in range(len(self.zmat.linx_indices)):
            for j in self.zmat.linx_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.linx_indices[i]
                    linx_carts = []
                    for k in indies[0]:
                        linx_carts = np.append(
                            linx_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    linx_carts = np.reshape(linx_carts, (-1, 3))
                    x1 = self.calc_Centroid(linx_carts)

                    linx_carts = []
                    for k in indies[1]:
                        linx_carts = np.append(
                            linx_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    linx_carts = np.reshape(linx_carts, (-1, 3))
                    x2 = self.calc_Centroid(linx_carts)

                    linx_carts = []
                    for k in indies[2]:
                        linx_carts = np.append(
                            linx_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    linx_carts = np.reshape(linx_carts, (-1, 3))
                    x3 = self.calc_Centroid(linx_carts)

                    linx_carts = []
                    for k in indies[3]:
                        linx_carts = np.append(
                            linx_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    linx_carts = np.reshape(linx_carts, (-1, 3))
                    x4 = self.calc_Centroid(linx_carts)
                else:
                    x1 = np.array(carts[int(self.zmat.linx_indices[i][0]) - 1]).astype(
                        float
                    )
                    x2 = np.array(carts[int(self.zmat.linx_indices[i][1]) - 1]).astype(
                        float
                    )
                    x3 = np.array(carts[int(self.zmat.linx_indices[i][2]) - 1]).astype(
                        float
                    )
                    x4 = np.array(carts[int(self.zmat.linx_indices[i][3]) - 1]).astype(
                        float
                    )
            lx = self.calc_Linx(x1, x2, x3, x4)
            int_coord = np.append(int_coord, lx)

        for i in range(len(self.zmat.liny_indices)):
            for j in self.zmat.liny_indices[i]:
                Len = len(np.array(j).shape)
                if Len:
                    indies = self.zmat.liny_indices[i]
                    liny_carts = []
                    for k in indies[0]:
                        liny_carts = np.append(
                            liny_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    liny_carts = np.reshape(liny_carts, (-1, 3))
                    x1 = self.calc_Centroid(liny_carts)

                    liny_carts = []
                    for k in indies[1]:
                        liny_carts = np.append(
                            liny_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    liny_carts = np.reshape(liny_carts, (-1, 3))
                    x2 = self.calc_Centroid(liny_carts)

                    liny_carts = []
                    for k in indies[2]:
                        liny_carts = np.append(
                            liny_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    liny_carts = np.reshape(liny_carts, (-1, 3))
                    x3 = self.calc_Centroid(liny_carts)

                    liny_carts = []
                    for k in indies[3]:
                        liny_carts = np.append(
                            liny_carts, np.array(carts[int(k) - 1]).astype(float)
                        )
                    liny_carts = np.reshape(liny_carts, (-1, 3))
                    x4 = self.calc_Centroid(liny_carts)
                else:
                    x1 = np.array(carts[int(self.zmat.liny_indices[i][0]) - 1]).astype(
                        float
                    )
                    x2 = np.array(carts[int(self.zmat.liny_indices[i][1]) - 1]).astype(
                        float
                    )
                    x3 = np.array(carts[int(self.zmat.liny_indices[i][2]) - 1]).astype(
                        float
                    )
                    x4 = np.array(carts[int(self.zmat.liny_indices[i][3]) - 1]).astype(
                        float
                    )
            ly = self.calc_Liny(x1, x2, x3, x4)
            int_coord = np.append(int_coord, ly)

        int_coord = np.dot(proj.T, int_coord)
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
        max_iter,
        tolerance,
        A,
        tight_disp,
        zmat,
        options,
    ):
        # tol = 1.0e-3
        # conv_iter = 10
        disp = n_disp.copy()
        new_n = n_coord + n_disp
        new_carts = np.array(ref_carts).astype(float)
        for i in range(max_iter):
            cart_disp = np.dot(n_disp, A)
            # if self.options.second_order:
            # A2 = self.compute_A2(self.s_vectors.B2, A)
            # A2 = np.einsum("ipq,pr,qs->irs", A2, self.ted.proj, self.ted.proj)
            # L = inv(self.eig_inv)
            # A2 = np.einsum("irs,rp,sq->ipq",A2, L, L)
            # cart_disp += 0.5*np.einsum("ipq,p,q->i", A2, n_disp,n_disp)
            cart_disp_shaped = np.reshape(cart_disp, (-1, 3))
            new_carts += cart_disp_shaped
            coord_check = self.int_c(new_carts, self.eig_inv, self.ted.proj)
            n_disp = new_n - coord_check
            # n_disp[np.abs(n_disp) < tol * np.max(np.abs(n_disp))] = 0

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
            # if self.options.reduced_disp:
            # disp = np.dot(disp, LA.inv(self.redu_norm_masses))
            print(np.where(abs(disp) == self.Disp))
            print("Norm:")
            print(LA.norm(n_disp))
            print("Tolerance:")
            print(tolerance)
            # raise RuntimeError
        return new_carts

    def compute_A(self, B, proj, eig_inv, u):
        """
        Construct 'A', the commented lines may be useful for getting
        intensities later. The BB^T product must be linearly independent
        to work, so it will be projected into the symmetry adapted basis.
        """

        L = inv(eig_inv)
        A = LA.pinv(B)  # (3N x s)
        A = np.dot(A, proj)  # (3N x S)
        A = A.T  # (S x 3N)

        # Alternative A-tensor computation
        # L = inv(eig_inv)
        # u = np.eye(np.shape(B)[1])
        # A = inv(B.dot(u).dot(B.T)) # (s x s)
        # A = (B.T).dot(A) (3N x s)
        # A = np.dot(A, proj)  # (3N x S)
        # A = A.T  # (S x 3N)

        # This could be necessary
        # for intensities.
        # A = A.dot(np.sqrt(inv(u)))
        # This step modifies A to convert from normal coords to carts.
        A = np.dot(L.T, A)  # (Q x 3N)

        return A

    def compute_A2(self, B2, A):

        C2 = np.einsum("rij,ip,jq->rpq", B2, A, A)

        A2 = np.einsum("rpq,ir->ipq", C2, A)

        return A2
