import numpy as np
import re
from numpy.linalg import inv
from numpy import linalg as LA


class FcRead:
    """
    Read a force-constant matrix from a formatted text file.

    This class parses force constants stored in the standard flattened
    three-column format commonly used by quantum chemistry programs and
    vibrational analysis utilities. All floating-point values are extracted
    from the file, assembled into a one-dimensional array, and reshaped into
    a square force-constant matrix.

    The matrix dimension is determined automatically from the total number
    of values read:

        N = sqrt(number of force constants)

    Parameters
    ----------
    file_name : str
        Name of the force-constant file to be read.

    Attributes
    ----------
    file_name : str
        Input filename.

    fc_regex : re.Pattern
        Regular expression used to extract floating-point values from the
        file.

    fc_mat : ndarray
        Square force-constant matrix reconstructed from the file contents.

    Notes
    -----
    The reader assumes that the file contains only the numerical elements
    of a square force-constant matrix written in row-major order. The total
    number of values must therefore be a perfect square.

    Examples
    --------
    >>> reader = FcRead("fc.dat")
    >>> reader.run()
    >>> F = reader.fc_mat
    >>> F.shape
    (54, 54)
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.fc_regex = re.compile(r"(-?\d+\.\d+)")

    def run(self):
        """
        Read and reconstruct the force-constant matrix.

        Opens the specified force-constant file, extracts all floating-point
        values, and reshapes them into a square NumPy array.

        Returns
        -------
        None

        Notes
        -----
        The resulting matrix is stored in the attribute ``fc_mat``.

        Raises
        ------
        ValueError
            If the number of extracted values cannot be reshaped into a
            square matrix.
        """
        with open(self.file_name, "r") as file:
            fc = file.read()

        self.fc_mat = re.findall(self.fc_regex, fc)
        self.fc_mat = np.reshape(
            self.fc_mat, (int(np.sqrt(len(self.fc_mat))), -1)
        ).astype(float)
