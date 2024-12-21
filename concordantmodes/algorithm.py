import numpy as np
import os
import scipy
from scipy import stats


class Algorithm(object):
    """
    The purpose of this class is to return a list of indices by which the force constants of the CMA method
    will be computed. These indices will be determined by user input or by a scoring function which takes into
    consideration the overlap of the normal coordinates and the difference in force constants for a particular normal
    mode.
    """

    def __init__(self, eigs, initial_fc, options):
        self.eigs = eigs
        self.options = options
        self.initial_fc = initial_fc

    def run(self):
        initial_fc = self.initial_fc

        tolerance = 5e-32
        a = self.eigs
        self.options.mode_coupling_check = False
        if self.options.mode_coupling_check:
            self.indices = self.coupling_diagnostic(a, initial_fc, tolerance)
        else:
            if self.options.off_diag:
                off_diag = self.options.off_diag_bands + 1
                if self.options.off_diag_limit != False:
                    lim = self.options.off_diag_limit - 1
                else:
                    lim = a + 1
            else:
                lim = a + 1
                off_diag = 1
            self.indices = self.loop(a, off_diag, lim)

    def loop(self, a, off_diag, lim):
        a = self.eigs
        indices = []
        Sum = 2
        for i in range(a):
            for j in range(i, i + off_diag):
                if j > a - 1:
                    break
                else:
                    if i == j:
                        indices.append([i, j])
                    elif i != j:
                        if i > lim:
                            break
                        else:
                            indices.append([i, j])
                    Sum += 2
        return indices

    def coupling_diagnostic(self, a, initial_fc, tolerance):
        indices = []
        diag = np.zeros((a, a))
        for x in range(a):
            for y in range(a):
                if x == y:
                    diag[x, x] = 0
                elif x != y:
                    diag[x, y] = S[x, y] / (initial_fc[x] - initial_fc[y])
        diag = np.absolute(diag)
        print(diag)
        data = np.abs(diag)
        hist, bin_edges = np.histogram(data, bins=a)
        for index, i in np.ndenumerate(data):
            if data[index] >= bin_edges[1]:
                indices.append(list(index))
        indices_new = []
        for index in indices:
            if index[1] > index[0]:
                indices_new.append(index)
        indices = indices_new

        return indices
