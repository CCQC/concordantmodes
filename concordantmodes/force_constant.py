import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class ForceConstant(object):
    # This script will calculate the force constants of the CMA normal
    # modes using numerical differentiation.

    # Symmetric First Derivative
    # [f(x+h) - f(x-h)] / 2*h =
    # [g_(i_plus) - g_(i_minus)] / 2*disp_size

    # Diagonal Second Derivative
    # [f(x+h) - 2f(x) + f(x-h)] / h^2 =
    # [E_(i_plus) - 2*E_(ref) + E_(i_minus)] / disp_size^2

    # Off-diagonal Second Derivative
    # [f(x+h,y+k) - f(x+h,y) - f(x,y+k) + 2f(x,y) - f(x-h,y) - f(x,y-k) + f(x-h,y-k) / 2*h^2
    # [E_(ij_plus) - E_(i_plus,j) - E_(i,j_plus) + E_(ref) - E_(i_minus,j) - E_(i,j_minus) + E_(i_minus,j_minus) / 2*disp_size^2

    def __init__(
        self,
        disp,
        p_array,
        m_array,
        ref_en,
        options,
        indices,
        deriv_level=0,
        coord_type_b="internal",
        cma_level="B",
    ):
        self.options = options
        self.disp = disp
        self.p_array = p_array
        self.m_array = m_array
        self.ref_en = ref_en
        self.indices = indices
        self.deriv_level = deriv_level
        self.coord_type_b = coord_type_b
        self.cma_level = cma_level

    def run(self):
        indices = self.indices
        disp = self.disp
        if self.coord_type_b == "cartesian":
            size = self.indices[-1][0] + 1
            cart_disp = np.zeros(size)
            for i in range(len(cart_disp)):
                cart_disp[i] = disp.disp_mag
            denom_disp = cart_disp
        else:
            denom_disp = self.disp.disp
        dim = self.p_array.shape[0]
        self.gradient = np.zeros((dim))
        self.FC = np.zeros((dim, dim))
        if not self.deriv_level:
            p_en_array = self.p_array
            m_en_array = self.m_array
            e_r = self.ref_en
            print("Force Constant disp sizes:")
            for index in indices:
                i, j = index[0], index[1]
                e_pi, e_pj = p_en_array[i, i], p_en_array[j, j]
                e_mi, e_mj = m_en_array[i, i], m_en_array[j, j]
                e_pp, e_mm = p_en_array[i, j], m_en_array[i, j]
                if i == j:
                    self.FC[i, i] = self.diag_fc(e_pi, e_mi, e_r, denom_disp[i])
                    self.gradient[i] = self.first_deriv(e_pi, e_mi, denom_disp[i])
                elif i != j:
                    self.FC[i, j] = self.off_diag_fc(
                        e_pp,
                        e_pi,
                        e_pj,
                        e_mi,
                        e_mj,
                        e_mm,
                        e_r,
                        denom_disp[i],
                        denom_disp[j],
                    )
                # if self.cma_level != "A":
                # e_pi, e_pj = p_en_array[i, i], p_en_array[j, j]
                # e_mi, e_mj = m_en_array[i, i], m_en_array[j, j]
                # e_pp, e_mm = p_en_array[i, j], m_en_array[i, j]
                # if i == j:
                # self.FC[i, i] = self.diag_fc(e_pi, e_mi, e_r, denom_disp[i])
                # self.gradient[i] = self.first_deriv(e_pi, e_mi, denom_disp[i])
                # elif i != j:
                # self.FC[i, j] = self.off_diag_fc(
                # e_pp,
                # e_pi,
                # e_pj,
                # e_mi,
                # e_mj,
                # e_mm,
                # e_r,
                # denom_disp[i],
                # denom_disp[j],
                # )
                # else:
                # e_pi, e_mi = p_en_array[i,j], m_en_array[i,j]
                # print("denom # "+str(i+1))
                # print(denom_disp[i])
                # self.FC[i, i] = self.diag_fc(e_pi, e_mi, e_r, denom_disp[i])
                # self.gradient[i] = self.first_deriv(e_pi, e_mi, denom_disp[i])
            # Take advantage of FC[i,j] = FC[j,i]
            cf = np.triu_indices(dim, 1)
            il = (cf[1], cf[0])
            self.FC[il] = self.FC[cf]
        elif self.deriv_level == 1:
            self.FC = (self.p_array - self.m_array) / (2 * denom_disp[0])
        else:
            print("Higher order deriv_level computations aren't yet supported")
            raise RuntimeError

    # I might want to shift the above computation in the deriv_level == 1 down here, though it is only one line
    def first_deriv(self, e_p, e_m, disp):
        return (e_p - e_m) / (2 * disp)

    # Functions for computing the diagonal and off-diagonal force constants
    def diag_fc(self, e_p, e_m, e_r, disp):
        fc = (e_p - 2 * e_r + e_m) / (disp**2)
        return fc

    def off_diag_fc(self, e_pi_pj, e_pi, e_pj, e_mi, e_mj, e_mi_mj, e_r, disp1, disp2):
        fc = (e_pi_pj - e_pi - e_pj + 2 * e_r - e_mi - e_mj + e_mi_mj) / (
            2 * disp1 * disp2
        )
        return fc
