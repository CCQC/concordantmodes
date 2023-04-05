import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power


class GFMethod(object):
    """
    This class is a versatile method to compute GF frequencies.

    TODO: Insert standard uncertainties of amu_elmass and HARTREE_WAVENUM
    """

    def __init__(self, G, F, tol, proj_tol, zmat, ted, cma=None):
        self.G = G
        self.F = F
        self.tol = tol
        self.proj_tol = proj_tol
        self.zmat = zmat
        self.AMU_ELMASS = 5.48579909065 * (10 ** (-4))
        self.HARTREE_WAVENUM = 219474.6313708
        self.ted = ted
        self.cma = cma

    def run(self):
        # Construct the orthogonalizer
        self.G_O = fractional_matrix_power(self.G, 0.5)

        # Symmetrize F, diagonalize, then backtransform the eigenvectors.
        self.F_O = np.dot(np.dot(self.G_O, self.F), self.G_O)
        self.eig_v, self.L_p = LA.eigh(self.F_O)
        self.L_p[np.abs(self.L_p) < self.tol] = 0
        self.L = np.dot(self.G_O, self.L_p)
        self.L = np.real(self.L)
        L = np.absolute(self.L)
        L_p = np.real(self.L_p)
        # S_p = np.dot(np.absolute(LA.inv(L_p)), np.absolute(L_p))
        self.L_p = L_p
        # make sure the correct "primed" or unprimed version is being saved in each instance
        # if self.cma:
        # with open("L_full.npy", "wb") as z:
        # np.save(z, L)
        # # if self.cma == False:
        # #    with open("L_intermediate.npy", "wb") as z:
        # #        np.save(z, L)
        # if self.cma is None:
        # with open("L_0.npy", "wb") as z:
        # np.save(z, L)
        # if self.cma == "init":
        # with open("S_p.npy", "wb") as z:
        # np.save(z, S_p)
        # Construct the normal mode overlap matrix. Will be useful for off-diagonal diagnostics.
        L = np.absolute(np.real(self.L))
        L_inv = LA.inv(L)
        S = np.dot(L.T, L)
        self.S = S

        self.L[np.abs(self.L) < self.tol] = 0
        # Compute the frequencies by the square root of the eigenvalues.
        self.freq = np.sqrt(self.eig_v, dtype=complex)
        # Filter for imaginary modes.
        for i in range(len(self.freq)):
            if np.real(self.freq[i]) > 0.0:
                self.freq[i] = np.real(self.freq[i])
            else:
                self.freq[i] = -np.imag(self.freq[i])

        # # This seems to fix the constant warning I get about casting out imaginary values from the complex numbers,
        # # however I will need to test it on transition states to see if it captures the imaginary frequencies.
        # self.freq = np.real(self.freq)
        # print(self.freq)
        self.freq = self.freq.astype(float)
        # Convert from Hartrees to wavenumbers.
        self.freq *= self.HARTREE_WAVENUM
        for i in range(len(self.freq)):
            print(
                "frequency #"
                + "{:3d}".format(i + 1)
                + ": "
                + "{:10.2f}".format(self.freq[i])
            )
        # Compute and then print the TED.
        # np.set_printoptions(suppress=True)
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format("Total Energy Distribution (TED)"))
        print("////////////////////////////////////////////")
        self.ted.run(self.L, self.freq, rect_print=False)
        self.ted_breakdown = self.ted.ted_breakdown
