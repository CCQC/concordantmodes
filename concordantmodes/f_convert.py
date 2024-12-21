import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class FcConv(object):
    """
    This class may be used to convert a cartesian F-matrix to an internal
    F-Matrix, and vice versa.
    This constant is from:
    https://physics.nist.gov/cgi-bin/cuu/Value?hr
    If this link dies, find the new link on NIST
    for the Hartree to Joule conversion and pop
    it in there.
    MDYNE_HART: Standard uncertainty of 0.0000000000085
    BOHR_ANG: Standard uncertainty of 0.00000000080
    """

    def __init__(self, fc_mat, s_vec, zmat, coord, print_f, ted, units, second_order):
        self.coord = coord
        # self.F_read = F_read
        self.F = fc_mat
        self.second_order = second_order
        self.print_f = print_f
        self.s_vec = s_vec
        self.ted = ted
        self.units = units
        self.zmat = zmat

        self.MDYNE_HART = 4.3597447222071
        self.BOHR_ANG = 0.529177210903

    def run(self, grad=np.array([])):
        # First construct the transpose of the A matrix.
        if self.coord.lower() == "internal":
            # The cartesian force constants must be in units of Hartree/bohr^2.
            if self.ted.proj is None:
                B = self.s_vec.B
            else:
                B = np.dot(self.ted.proj.T, self.s_vec.B)
            BT = B.T
            G = np.dot(B, BT)
            self.A_T = np.dot(LA.inv(G), B)
            if self.ted.proj is not None:
                self.A_T = np.dot(self.ted.proj, self.A_T)
            if self.units == "MdyneAng":
                self.F /= self.BOHR_ANG
                self.F *= self.MDYNE_HART
            self.F = np.einsum("pi,rj,ij->pr", self.A_T, self.A_T, self.F)
            # Non-stationary, gradient correction to internal coordinate force constants
            V2 = self.F.copy() * 0
            if len(grad) and self.second_order:
                # Note: I may want to use projected A_T to reduce the size of the
                # internal coordinate basis, thus speeding up the computation.
                # The basis will eventually need to be projected anyways.
                np.set_printoptions(precision=6, linewidth=240)
                self.v_q = np.dot(self.A_T, grad)
                C2 = np.einsum("rij,pi,qj->rpq", self.s_vec.B2, self.A_T, self.A_T)
                V2 = np.einsum("q,qpr->pr", self.v_q, C2)

            self.F = self.F - V2

            if self.print_f:
                self.print_const()
        elif self.coord.lower() == "cartesian":
            self.F = np.einsum("pi,rj,pr->ij", self.s_vec.B, self.s_vec.B, self.F)
            if self.print_f:
                self.print_const()

    def print_const(self, fc_name="fcFinal.dat"):
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
