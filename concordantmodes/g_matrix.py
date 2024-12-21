import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class GMatrix(object):
    """
    This class will be used to compute the G-matrix of the GF-Matrix Method.
    It should be constructed to give the flexibility of transformation via
    the L-matrix.
    """

    def __init__(self, zmat, s_vectors, options):
        self.zmat = zmat
        self.s_vectors = s_vectors
        self.options = options

    def run(self):
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
