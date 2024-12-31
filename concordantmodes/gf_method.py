import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
import copy

class GFMethod(object):
    """
    This class is a versatile method to compute GF frequencies.

    TODO: Insert standard uncertainties of amu_elmass and HARTREE_WAVENUM
    """

    def __init__(self, G, F, tol, proj_tol, zmat, ted, options, symtext = None, cma=None, sym_sort=[]):
        self.G = G
        self.F = F
        self.tol = tol
        self.proj_tol = proj_tol
        self.zmat = zmat
        self.AMU_ELMASS = 5.48579909065 * (10 ** (-4))
        self.HARTREE_WAVENUM = 219474.6313708
        self.ted = ted
        self.symtext = symtext
        self.cma = cma
        self.options = options

    def run(self):
        # Construct the orthogonalizer
        self.G_O = fractional_matrix_power(self.G, 0.5)

        # Symmetrize F, diagonalize, then backtransform the eigenvectors.
        self.F_O = np.dot(np.dot(self.G_O, self.F), self.G_O)
   
        if self.options.symmetry:
            #allows the block diagonalization of the GF matrix so normal modes stay
            #in the assumed ordering for the level A displacements (Within their symmetry blocks)
            self.block_GF()
        else:
            self.eig_v, self.L_p = LA.eigh(self.F_O)
        
        self.L_p[np.abs(self.L_p) < self.tol] = 0
        self.L = np.dot(self.G_O, self.L_p)
        self.L = np.real(self.L)
        L = np.absolute(self.L)
        L_p = np.real(self.L_p)
        self.L_p = L_p

        self.L[np.abs(self.L) < self.tol] = 0
        # Compute the frequencies by the square root of the eigenvalues.
        self.freq = np.sqrt(self.eig_v, dtype=complex)
        # Filter for imaginary modes.
        for i in range(len(self.freq)):
            if np.real(self.freq[i]) > 0.0:
                self.freq[i] = np.real(self.freq[i])
            else:
                self.freq[i] = -np.imag(self.freq[i])

        # This seems to fix the constant warning I get about casting out imaginary values from the complex numbers,
        # however I will need to test it on transition states to see if it captures the imaginary frequencies.
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
        print("////////////////////////////////////////////")
        print("//{:^40s}//".format("Total Energy Distribution (TED)"))
        print("////////////////////////////////////////////")
        if self.options.symmetry:
            self.ted.run(self.L, self.freq, self.symtext, rect_print=False)
        else:
            self.ted.run(self.L, self.freq, rect_print=False)
        self.ted_breakdown = self.ted.ted_breakdown

    def block_GF(self):
        self.eigvals, self.eigvecs = [], []
        offset_h = 0
        for hi, h in enumerate(self.symtext.salcblocks):
            
            #Extract the block from the supermatrix
            fo = self.extract(self.F_O, offset_h, h.shape[0])
            eig_v_h, L_p_h = LA.eigh(fo)
            self.eigvals.append(eig_v_h)
            self.eigvecs.append(L_p_h)

            # UPDATE OFFSET
            offset_h += h.shape[0] 
        self.eigvals = [item for sublist in self.eigvals for item in sublist]
        self.eig_v = np.asarray(self.eigvals)
        self.L_p = block_diag(*self.eigvecs)
  
     #function to extract bd matrices from supermatrix
    def extract(self, A, offset, size):
        return A[offset:offset+size, offset:offset+size]