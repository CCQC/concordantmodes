import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import logging
from concordantmodes.log import debug_print, info_print

logger = logging.getLogger(__name__)


class OffDiagonal:
    """
    This class handles all off-diagonal CMA methods.
    """

    def __init__(self, options, f_diag, od_inds=[]):
        self.options = options
        self.od_inds = od_inds
        self.f_diag = f_diag

    def run(self):
        off_diag = self.options.off_diag
        od_inds = self.od_inds
        temp = copy.copy(f_diag)
        if off_diag == 1:
            debug_print(logger, "Adding on these off-diagonals:")
            debug_print(logger, od_inds)
