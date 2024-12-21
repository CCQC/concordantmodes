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
        anharm=False,
        anharm_indices=[],
        p_anharm=[],
        m_anharm=[],
        coord_type_init="internal",
    ):
        self.options = options
        self.disp = disp
        self.p_array = p_array
        self.m_array = m_array
        self.ref_en = ref_en
        self.indices = indices
        self.deriv_level = deriv_level
        self.anharm = anharm
        self.anharm_indices = anharm_indices
        self.p_anharm = p_anharm
        self.m_anharm = m_anharm
        self.coord_type_init = coord_type_init

    def run(self):
        indices = self.indices
        disp = self.disp
        if self.coord_type_init == "cartesian":
            size = self.indices[-1][0] + 1
            cart_disp = np.zeros(size)
            for i in range(len(cart_disp)):
                cart_disp[i] = disp.disp
            disp.disp = cart_disp
        dim = self.p_array.shape[0]
        self.FC = np.zeros((dim, dim))
        if not self.deriv_level:
            if not self.anharm:
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
                            e_pp,
                            e_pi,
                            e_pj,
                            e_mi,
                            e_mj,
                            e_mm,
                            e_r,
                            disp.disp[i],
                            disp.disp[j],
                        )
                # Take advantage of FC[i,j] = FC[j,i]
                cf = np.triu_indices(dim, 1)
                il = (cf[1], cf[0])
                self.FC[il] = self.FC[cf]
            else:
                p_en_array = self.p_array
                m_en_array = self.m_array
                p_anharm = self.p_anharm
                m_anharm = self.m_anharm
                e_r = self.ref_en
                self.f_cube = np.zeros((dim, dim, dim))
                for i in range(len(self.anharm_indices[0])):
                    self.f_cube[i, i, i] = self.cubic_fc_diag(
                        p_anharm[i],
                        p_en_array[i, i],
                        m_anharm[i],
                        m_en_array[i, i],
                        self.disp.disp[i],
                    )
                l = len(self.anharm_indices[0])
                for i in range(len(self.anharm_indices[1])):
                    a = self.anharm_indices[1][i][0]
                    b = self.anharm_indices[1][i][2]
                    self.f_cube[a, a, b] = self.cubic_fc_f_xxy(
                        p_anharm[i + l],
                        m_anharm[i + l],
                        p_en_array[a, a],
                        m_en_array[a, a],
                        p_en_array[b, b],
                        m_en_array[b, b],
                        self.disp.disp[a],
                        self.disp.disp[b],
                    )
                    self.f_cube[a, b, a] = self.f_cube[a, a, b]
                    self.f_cube[b, a, a] = self.f_cube[a, a, b]
                l += len(self.anharm_indices[1])
                for i in range(len(self.anharm_indices[2])):
                    a = self.anharm_indices[2][i][0]
                    b = self.anharm_indices[2][i][1]
                    c = self.anharm_indices[2][i][2]
                    self.f_cube[a, b, c] = self.cubic_fc_f_xyz(
                        p_anharm[i + l],
                        p_en_array[a, a],
                        p_en_array[b, b],
                        p_en_array[c, c],
                        m_anharm[i + l],
                        m_en_array[a, a],
                        m_en_array[b, b],
                        m_en_array[c, c],
                        self.disp.disp[a],
                        self.disp.disp[b],
                        self.disp.disp[c],
                    )
                    self.f_cube[b, a, c] = self.f_cube[a, b, c]
                    self.f_cube[b, c, a] = self.f_cube[a, b, c]
                    self.f_cube[a, c, b] = self.f_cube[a, b, c]
                    self.f_cube[c, a, b] = self.f_cube[a, b, c]
                    self.f_cube[c, b, a] = self.f_cube[a, b, c]
                self.f_quart = np.zeros((dim, dim, dim, dim))
                l += len(self.anharm_indices[2])
                for i in range(len(self.anharm_indices[3])):
                    self.f_quart[i, i, i, i] = self.quartic_fc_diag(
                        p_anharm[i],
                        p_en_array[i, i],
                        m_anharm[i],
                        m_en_array[i, i],
                        e_r,
                        self.disp.disp[i],
                    )
                l += len(self.anharm_indices[3])
                for i in range(len(self.anharm_indices[4])):
                    a = self.anharm_indices[4][i][0]
                    b = self.anharm_indices[4][i][3]
                    self.f_quart[a, a, a, b] = self.quartic_fc_f_xxxy(
                        p_anharm[i + l],
                        p_anharm[a],
                        p_en_array[a, b],
                        p_en_array[a, a],
                        m_anharm[i + l],
                        m_anharm[a],
                        m_en_array[a, b],
                        m_en_array[a, a],
                        e_r,
                        self.disp.disp[a],
                        self.disp.disp[b],
                    )
                    self.f_quart[a, a, b, a] = self.f_quart[a, a, a, b]
                    self.f_quart[a, b, a, a] = self.f_quart[a, a, a, b]
                    self.f_quart[b, a, a, a] = self.f_quart[a, a, a, b]
                l += len(self.anharm_indices[4])
                for i in range(len(self.anharm_indices[5])):
                    a = self.anharm_indices[5][i][0]
                    b = self.anharm_indices[5][i][2]
                    self.f_quart[a, a, b, b] = self.quartic_fc_f_xxyy(
                        p_anharm[i + l],
                        p_anharm[a],
                        p_anharm[b],
                        p_en_array[a, b],
                        p_en_array[a, a],
                        p_en_array[b, b],
                        m_anharm[i + l],
                        m_anharm[a],
                        m_anharm[b],
                        m_en_array[a, b],
                        m_en_array[a, a],
                        m_en_array[b, b],
                        e_r,
                        self.disp.disp[a],
                        self.disp.disp[b],
                    )
                    self.f_quart[a, b, a, b] = self.f_quart[a, a, b, b]
                    self.f_quart[b, a, a, b] = self.f_quart[a, a, b, b]
                    self.f_quart[a, b, b, a] = self.f_quart[a, a, b, b]
                    self.f_quart[b, a, b, a] = self.f_quart[a, a, b, b]
                    self.f_quart[b, b, a, a] = self.f_quart[a, a, b, b]
                l += len(self.anharm_indices[5])
                for i in range(len(self.anharm_indices[6])):
                    a = self.anharm_indices[6][i][0]
                    b = self.anharm_indices[6][i][2]
                    c = self.anharm_indices[6][i][3]
                    self.f_quart[a, a, b, c] = self.quartic_fc_f_xxyz(
                        p_anharm[i + l],
                        p_anharm[a],
                        p_en_array[a, b],
                        p_en_array[a, c],
                        p_en_array[b, c],
                        p_en_array[a, a],
                        m_anharm[i + l],
                        m_anharm[a],
                        m_en_array[a, b],
                        m_en_array[a, c],
                        m_en_array[b, c],
                        m_en_array[a, a],
                        e_r,
                        self.disp.disp[a],
                        self.disp.disp[b],
                        self.disp.disp[c],
                    )
                    self.f_quart[a, b, a, c] = self.f_quart[a, a, b, c]
                    self.f_quart[b, a, a, c] = self.f_quart[a, a, b, c]
                    self.f_quart[a, a, c, b] = self.f_quart[a, a, b, c]
                    self.f_quart[a, b, c, a] = self.f_quart[a, a, b, c]
                    self.f_quart[b, a, c, a] = self.f_quart[a, a, b, c]
                    self.f_quart[a, c, a, b] = self.f_quart[a, a, b, c]
                    self.f_quart[a, c, b, a] = self.f_quart[a, a, b, c]
                    self.f_quart[b, c, a, a] = self.f_quart[a, a, b, c]
                    self.f_quart[c, a, a, b] = self.f_quart[a, a, b, c]
                    self.f_quart[c, a, b, a] = self.f_quart[a, a, b, c]
                    self.f_quart[c, b, a, a] = self.f_quart[a, a, b, c]
                l += len(self.anharm_indices[6])
                for i in range(len(self.anharm_indices[7])):
                    a = self.anharm_indices[7][i][0]
                    b = self.anharm_indices[7][i][1]
                    c = self.anharm_indices[7][i][2]
                    d = self.anharm_indices[7][i][3]
                    self.f_quart[a, b, c, d] = self.quartic_fc_f_wxyz(
                        p_anharm[i + l],
                        p_en_array[a, b],
                        p_en_array[a, c],
                        p_en_array[a, d],
                        p_en_array[b, c],
                        p_en_array[b, d],
                        p_en_array[c, d],
                        m_anharm[i + l],
                        m_en_array[a, b],
                        m_en_array[a, c],
                        m_en_array[a, d],
                        m_en_array[b, c],
                        m_en_array[b, d],
                        m_en_array[c, d],
                        e_r,
                        self.disp.disp[a],
                        self.disp.disp[b],
                        self.disp.disp[c],
                        self.disp.disp[d],
                    )
                    self.f_quart[b, a, c, d] = self.f_quart[a, b, c, d]
                    self.f_quart[b, c, a, d] = self.f_quart[a, b, c, d]
                    self.f_quart[a, c, b, d] = self.f_quart[a, b, c, d]
                    self.f_quart[c, a, b, d] = self.f_quart[a, b, c, d]
                    self.f_quart[c, b, a, d] = self.f_quart[a, b, c, d]
                    self.f_quart[a, b, d, c] = self.f_quart[a, b, c, d]
                    self.f_quart[b, a, d, c] = self.f_quart[a, b, c, d]
                    self.f_quart[b, c, d, a] = self.f_quart[a, b, c, d]
                    self.f_quart[a, c, d, b] = self.f_quart[a, b, c, d]
                    self.f_quart[c, a, d, b] = self.f_quart[a, b, c, d]
                    self.f_quart[c, b, d, a] = self.f_quart[a, b, c, d]
                    self.f_quart[a, d, b, c] = self.f_quart[a, b, c, d]
                    self.f_quart[b, d, a, c] = self.f_quart[a, b, c, d]
                    self.f_quart[b, d, c, a] = self.f_quart[a, b, c, d]
                    self.f_quart[a, d, c, b] = self.f_quart[a, b, c, d]
                    self.f_quart[c, d, a, b] = self.f_quart[a, b, c, d]
                    self.f_quart[c, d, b, a] = self.f_quart[a, b, c, d]
                    self.f_quart[d, a, b, c] = self.f_quart[a, b, c, d]
                    self.f_quart[d, b, a, c] = self.f_quart[a, b, c, d]
                    self.f_quart[d, b, c, a] = self.f_quart[a, b, c, d]
                    self.f_quart[d, a, c, b] = self.f_quart[a, b, c, d]
                    self.f_quart[d, c, a, b] = self.f_quart[a, b, c, d]
                    self.f_quart[d, c, b, a] = self.f_quart[a, b, c, d]
        elif self.deriv_level == 1:
            self.FC = (self.p_array - self.m_array) / (2 * self.disp.disp[0])
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

    def off_diag_fc(self, e_pi_pj, e_pi, e_pj, e_mi, e_mj, e_mi_mj, e_r, disp1, disp2):
        fc = (e_pi_pj - e_pi - e_pj + 2 * e_r - e_mi - e_mj + e_mi_mj) / (
            2 * disp1 * disp2
        )
        return fc

    def cubic_fc_diag(self, e_pp, e_p, e_mm, e_m, disp):
        fc = (0.5 * e_pp - e_p + e_m - 0.5 * e_mm) / (disp**3)
        return fc

    # Only 2 extra terms from cubic_fc_diag
    def cubic_fc_f_xxy(self, e_ppi_pj, e_mmi_mj, e_pi, e_mi, e_pj, e_mj, disp1, disp2):
        fc = (e_ppi_pj - e_mmi_mj - 2 * e_pi + 2 * e_mi - e_pj + e_mj) / (
            4 * disp2 * disp1**2
        )
        return fc

    # Only 2 extra terms from cubic_fc_diag
    def cubic_fc_f_xyz(
        self,
        e_pi_pj_pk,
        e_pi,
        e_pj,
        e_pk,
        e_mi_mj_mk,
        e_mi,
        e_mj,
        e_mk,
        disp1,
        disp2,
        disp3,
    ):
        fc = (e_pi_pj_pk - e_mi_mj_mk - e_pi + e_mi - e_pj + e_mj - e_pk + e_mk) / (
            disp1 * disp2 * disp3
        )
        return fc

    def quartic_fc_diag(self, e_pp, e_p, e_mm, e_m, e_r, disp):
        fc = (e_pp - 4 * e_p + 6 * e_r - 4 * e_m + e_mm) / (disp**4)
        return fc

    def quartic_fc_f_xxxy(
        self,
        e_pppi_pj,
        e_ppi,
        e_pi_pj,
        e_pi,
        e_mmmi_mj,
        e_mmi,
        e_mi_mj,
        e_mi,
        e_r,
        disp1,
        disp2,
    ):
        fc = (
            (
                e_pppi_pj
                + e_mmmi_mj
                - 9 * (e_pi + e_mi)
                - 3 * (e_pi_pj - e_mi_mj)
                + (3 / 4) * (e_ppi + e_mmi)
                + (41 / 2) * e_r
            )
            / 8
            * (disp1**3 * disp2)
        )
        return fc

    def quartic_fc_f_xxyy(
        self,
        e_ppi_ppj,
        e_ppi,
        e_ppj,
        e_pi_pj,
        e_pi,
        e_pj,
        e_mmi_mmj,
        e_mmi,
        e_mmj,
        e_mi_mj,
        e_mi,
        e_mj,
        e_r,
        disp1,
        disp2,
    ):
        fc = (
            (
                e_ppi_ppj
                + e_mmi_mmj
                - e_ppi
                - e_mmi
                - e_ppj
                - e_mmj
                - 4 * (e_pi_pj + e_mi_mj)
                + 4 * (e_pi + e_mi)
                + 4 * (e_pj + e_mj)
                - 6 * e_r
            )
            / 6
            * (disp1**2 * disp2**2)
        )
        return fc

    def quartic_fc_f_xxyz(
        self,
        e_ppi_pj_pk,
        e_ppi,
        e_pi_pj,
        e_pi_pk,
        e_pj_pk,
        e_pi,
        e_mmi_mj_mk,
        e_mmi,
        e_mi_mj,
        e_mi_mk,
        e_mj_mk,
        e_mi,
        e_r,
        disp1,
        disp2,
        disp3,
    ):
        fc = (
            (
                e_ppi_pj_pk
                + e_mmi_mj_mk
                - e_ppi
                - e_mmi
                - 2 * (e_pi_pj + e_mi_mj)
                - 2 * (e_pi_pk + e_mi_mk)
                - e_pj_pk
                - e_mj_mk
                + 4 * (e_pi + e_mi)
                + 2 * e_r
            )
            / 4
            * (disp1**2 * disp2 * disp3)
        )
        return fc

    def quartic_fc_f_wxyz(
        self,
        e_pi_pj_pk_pl,
        e_pi_pj,
        e_pi_pk,
        e_pi_pl,
        e_pj_pk,
        e_pj_pl,
        e_pk_pl,
        e_mi_mj_mk_ml,
        e_mi_mj,
        e_mi_mk,
        e_mi_ml,
        e_mj_mk,
        e_mj_ml,
        e_mk_ml,
        e_r,
        disp1,
        disp2,
        disp3,
        disp4,
    ):
        fc = (
            (
                e_pi_pj_pk_pl
                + e_mi_mj_mk_ml
                - e_pi_pj
                - e_mi_mj
                - e_pi_pk
                - e_mi_mk
                - e_pi_pl
                - e_mi_ml
                - e_pj_pk
                - e_mj_mk
                - e_pj_pl
                - e_mj_ml
                - e_pk_pl
                - e_mk_ml
                + 10 * e_r
            )
            / 2
            * (disp1 * disp2 * disp3 * disp4)
        )
        return fc
