import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class ForceConstant:
    """
    Compute force constants by finite differentiation of energies or gradients.

    This class constructs harmonic force-constant matrices in the coordinate
    basis defined by the displacement generator. Force constants are obtained
    using central finite-difference formulas applied to electronic energies
    or gradients evaluated at displaced geometries.

    Two differentiation schemes are supported:

    * Energy finite differences (``deriv_level=0``)
      - Computes first derivatives (gradients) and second derivatives
        (force constants) from displaced energies.
      - Includes diagonal and off-diagonal Hessian elements.

    * Gradient finite differences (``deriv_level=1``)
      - Computes force constants directly from displaced gradients.
      - Produces a symmetric Hessian matrix.

    The resulting force-constant matrix is expressed in the displacement
    coordinate basis used by the Concordant Modes Algorithm (CMA), which may
    correspond to projected internal coordinates, normal coordinates, or
    Cartesian coordinates depending on the calculation setup.

    Parameters
    ----------
    disp : TransfDisp
        Displacement object containing displacement magnitudes and
        coordinate transformation information.

    p_array : ndarray
        Energies or gradients evaluated at positive displacements.

    m_array : ndarray
        Energies or gradients evaluated at negative displacements.

    ref_en : float or None
        Reference energy at the equilibrium geometry. Required for
        energy-based finite differences and ignored for gradient-based
        calculations.

    options : object
        User-defined options controlling the CMA calculation.

    indices : list[list[int]]
        Coordinate index pairs defining which force constants are to be
        evaluated.

    deriv_level : int, optional
        Differentiation level used to construct the Hessian.

        - ``0`` : Hessian from energies.
        - ``1`` : Hessian from gradients.

    coord_type : {"internal", "cartesian"}, optional
        Coordinate system in which the displacements were generated.

    cma_level : {"A", "B", "C"}, optional
        CMA stage associated with the force-constant calculation.

    gradient : ndarray, optional
        Reference gradient vector. Primarily used for higher-order
        coordinate transformations and non-stationary reference points.

    Attributes
    ----------
    FC : ndarray
        Computed force-constant matrix.

    gradient : ndarray
        Gradient vector computed from central finite differences when
        ``deriv_level=0``.

    Notes
    -----
    The following central finite-difference formulas are employed.

    First derivative:

    .. math::

        \\frac{df}{dq_i}
        =
        \\frac{f(q_i+h)-f(q_i-h)}
             {2h}

    Diagonal force constant:

    .. math::

        \\frac{\\partial^2 E}{\\partial q_i^2}
        =
        \\frac{E(q_i+h)-2E_0+E(q_i-h)}
             {h^2}

    Off-diagonal force constant:

    .. math::

        \\frac{\\partial^2 E}
             {\\partial q_i \\partial q_j}
        =
        \\frac{
        E(+i,+j)-E(+i)-E(+j)
        +2E_0
        -E(-i)-E(-j)
        +E(-i,-j)}
        {2h_i h_j}

    Symmetry of the Hessian is enforced after construction:

    .. math::

        F_{ij} = F_{ji}

    for all coordinate pairs.
    """
    
    def __init__(
        self,
        disp,
        p_array,
        m_array,
        ref_en,
        options,
        indices,
        deriv_level=0,
        coord_type="internal",
        cma_level="B",
        gradient=[],
    ):
        self.options = options
        self.disp = disp
        self.p_array = p_array
        self.m_array = m_array
        self.ref_en = ref_en
        self.indices = indices
        self.deriv_level = deriv_level
        self.coord_type = coord_type
        self.cma_level = cma_level
        self.gradient = gradient

    def run(self):
        """
        Construct the force-constant matrix.

        Depending on the selected derivative level, computes force constants
        from either displaced energies or displaced gradients. For energy-based
        calculations, both gradients and Hessian elements are evaluated using
        central finite differences. For gradient-based calculations, the Hessian
        is obtained directly from gradient differences and subsequently
        symmetrized.

        Returns
        -------
        None

        Notes
        -----
        The computed force-constant matrix is stored in ``self.FC``. For
        energy-based calculations, the corresponding finite-difference
        gradient is stored in ``self.gradient``.
        """
        indices = self.indices
        disp = self.disp
        if self.coord_type == "cartesian":
            size = self.indices[-1][0] + 1
            cart_disp = np.zeros(size)
            for i in range(len(cart_disp)):
                cart_disp[i] = disp.disp_mag
            denom_disp = cart_disp
        else:
            denom_disp = self.disp.disp
        dim = self.p_array.shape[0]
        self.FC = np.zeros((dim, dim))
        if not self.deriv_level:
            self.gradient = np.zeros((dim))
            p_en_array = self.p_array
            m_en_array = self.m_array
            e_r = self.ref_en
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
            # Take advantage of FC[i,j] = FC[j,i]
            cf = np.triu_indices(dim, 1)
            il = (cf[1], cf[0])
            self.FC[il] = self.FC[cf]
        elif self.deriv_level == 1:
            self.FC = self.first_deriv(self.p_array, self.m_array, denom_disp[0])
            for i in range(len(self.FC) - 1):
                for j in range(i + 1):
                    self.FC[i + 1, j] = (self.FC[i + 1, j] + self.FC[j, i + 1]) / 2
                    self.FC[j, i + 1] = self.FC[i + 1, j]
        else:
            print("Higher order deriv_level computations aren't yet supported")
            raise RuntimeError

    # I might want to shift the above computation in the deriv_level == 1 down here, though it is only one line
    def first_deriv(self, e_p, e_m, disp):
        """
        Compute a first derivative using a central finite difference.

        Parameters
        ----------
        e_p : float or ndarray
            Quantity evaluated at the positive displacement.

        e_m : float or ndarray
            Quantity evaluated at the negative displacement.

        disp : float
            Displacement magnitude.

        Returns
        -------
        float or ndarray
            First derivative estimate.
        """
        return (e_p - e_m) / (2 * disp)

    # Functions for computing the diagonal and off-diagonal force constants
    def diag_fc(self, e_p, e_m, e_r, disp):
        """
        Compute a diagonal Hessian element by finite differences.

        Parameters
        ----------
        e_p, e_m : float
            Energies at positive and negative displacements.

        e_r : float
            Reference energy.

        disp : float
            Displacement magnitude.

        Returns
        -------
        float
            Diagonal force constant.
        """
        fc = (e_p - 2 * e_r + e_m) / (disp**2)
        return fc

    def off_diag_fc(self, e_pi_pj, e_pi, e_pj, e_mi, e_mj, e_mi_mj, e_r, disp1, disp2):
        """
        Compute an off-diagonal Hessian element by finite differences.

        Parameters
        ----------
        e_pi_pj : float
            Energy at the simultaneous positive displacement of coordinates
            ``i`` and ``j``.

        e_pi, e_pj : float
            Energies at individual positive displacements.

        e_mi, e_mj : float
            Energies at individual negative displacements.

        e_mi_mj : float
            Energy at the simultaneous negative displacement of coordinates
            ``i`` and ``j``.

        e_r : float
            Reference energy.

        disp1, disp2 : float
            Displacement magnitudes for coordinates ``i`` and ``j``.

        Returns
        -------
        float
            Off-diagonal force constant.
        """
        fc = (e_pi_pj - e_pi - e_pj + 2 * e_r - e_mi - e_mj + e_mi_mj) / (
            2 * disp1 * disp2
        )
        return fc
