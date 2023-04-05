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

    def __init__(self, disp, p_array, m_array, ref_en, options, indices, deriv_level=0):
        self.options = options
        self.disp = disp
        self.p_array = p_array
        self.m_array = m_array
        self.ref_en = ref_en
        self.indices = indices
        self.deriv_level = deriv_level

    def run(self):
        indices = self.indices
        disp = self.disp
        a = self.p_array.shape[0]
        self.FC = np.zeros((a, a))
        if not self.deriv_level:
            p_en_array = self.p_array
            m_en_array = self.m_array
            e_r = self.ref_en
            for index in indices:
                i, j = index[0], index[1]
                e_pi, e_pj = p_en_array[i, i], p_en_array[j, j]
                e_mi, e_mj = m_en_array[i, i], m_en_array[j, j]
                e_pp, e_mm = p_en_array[i, j], m_en_array[i, j]
                if i == j:
                    self.FC[i, i] = self.diag_fc(e_pi, e_mi, e_r, disp.disp[i])
                elif i != j:
                    self.FC[i, j] = self.off_diag_fc(
                        e_pp, e_pi, e_pj, e_mi, e_mj, e_mm, e_r, disp.disp[i]
                    )
            # Take advantage of FC[i,j] = FC[j,i]
            cf = np.triu_indices(a, 1)
            il = (cf[1], cf[0])
            self.FC[il] = self.FC[cf]
        elif self.deriv_level == 1:
            self.FC = (self.p_array - self.m_array) / (2 * self.disp.disp[0])
            # raise RuntimeError
        else:
            print("Higher order deriv_level computations aren't yet supported")
            raise RuntimeError

    # I might want to shift the above computation in the deriv_level == 1 down here, though it is only one line
    def first_deriv(self):
        pass

    # Functions for computing the diagonal and off-diagonal force constants
    def diag_fc(self, e_p, e_m, e_r, disp):
        fc = (e_p - 2 * e_r + e_m) / (disp**2)
        return fc

    def off_diag_fc(self, e_pp, e_pi, e_pj, e_mi, e_mj, e_mm, e_r, disp):
        fc = (e_pp - e_pi - e_pj + 2 * e_r - e_mi - e_mj + e_mm) / (2 * (disp**2))
        return fc
