import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class GMatrix:
    """
    Construct the Wilson G-matrix for vibrational analysis.

    This class computes the kinetic energy matrix (G-matrix) used in the
    Wilson GF method for determining molecular vibrational frequencies and
    normal modes. The G-matrix is constructed from the Wilson B-matrix and
    the inverse atomic mass matrix according to

    .. math::

        G = B M^{-1} B^T

    where:

    - ``B`` is the Wilson B-matrix relating internal-coordinate
      displacements to Cartesian displacements.
    - ``M`` is the diagonal Cartesian mass matrix.

    The resulting G-matrix represents the kinetic energy metric in the
    internal-coordinate basis and is subsequently combined with the
    force-constant matrix in the generalized eigenvalue problem

    .. math::

        GF L = L \\Lambda

    to obtain harmonic vibrational frequencies and normal coordinates.

    The class also supports projection into a reduced coordinate space,
    such as delocalized internal coordinates or projected vibrational
    coordinates used in the Concordant Modes Algorithm (CMA).

    Parameters
    ----------
    zmat : Zmat
        Molecular coordinate object containing atomic masses, atom labels,
        and internal-coordinate definitions.

    s_vectors : SVectors
        Object containing the Wilson B-matrix and related internal-coordinate
        transformation information.

    options : object
        User-defined options controlling coordinate-system selection and
        projection behavior.

    proj : ndarray, optional
        Projection matrix defining a reduced vibrational coordinate basis.
        If provided, the G-matrix is transformed into the projected space.

    Attributes
    ----------
    G : ndarray
        Wilson G-matrix in either the full internal-coordinate basis or
        the projected coordinate basis.

    zmat : Zmat
        Molecular coordinate representation.

    s_vectors : SVectors
        Internal-coordinate derivative object containing the Wilson
        B-matrix.

    proj : ndarray
        Projection matrix used to reduce the coordinate basis.

    Notes
    -----
    Atomic masses are converted to inverse masses prior to constructing
    the Cartesian mass matrix. Dummy atoms labeled ``"X"`` are assigned
    zero inverse mass so that they do not contribute to the kinetic energy.

    Small numerical elements are removed in two stages:

    1. Absolute thresholding to eliminate numerical noise.
    2. Relative thresholding based on the largest value in each row.

    This improves numerical stability for subsequent GF matrix
    diagonalizations.
    """

    def __init__(self, zmat, s_vectors, options, proj=np.array([])):
        self.zmat = zmat
        self.s_vectors = s_vectors
        self.options = options
        self.proj = proj

    def run(self):
        """
        Compute the Wilson G-matrix.

        Constructs the inverse Cartesian mass matrix from the atomic masses,
        evaluates

        .. math::

            G = B M^{-1} B^T

        and optionally projects the result into a reduced coordinate basis.

        Small numerical elements are removed to improve numerical stability
        during subsequent diagonalization of the GF matrix.

        Returns
        -------
        None

        Notes
        -----
        The computed matrix is stored in the instance attribute ``G``.
        If a projection matrix is supplied and the coordinate system is not
        a conventional Z-matrix representation, the projected matrix

        .. math::

            G' = P^T G P

        is formed and stored instead.
        """
        B = self.s_vectors.B

        # Compute and temper G
        tol = 1e-12
        u = np.array(self.zmat.masses)
        for i in range(len(u)):
            if self.zmat.atom_list[i] == "X":
                u[i] = 0
            else:
                u[i] = 1.0 / u[i]
        u = np.repeat(u, 3)
        u = np.diag(u)
        self.G = B.dot(u.dot(B.T))
        self.G[np.abs(self.G) < tol] = 0

        if len(self.proj) and self.options.coords != "ZMAT":
            self.G = np.dot(self.proj.T, np.dot(self.G, self.proj))

        del_tol = 1.0e-3
        for row in self.G:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row) * del_tol] = 0
        self.G = self.G.T
        for row in self.G:
            abs_row = np.abs(row)
            row[abs_row < np.max(abs_row) * del_tol] = 0
        self.G = self.G.T
