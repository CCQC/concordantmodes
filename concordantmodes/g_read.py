import numpy as np
import re
from numpy.linalg import inv
from numpy import linalg as LA


class GrRead:
    """
    Read Cartesian gradient data from a formatted gradient file.

    This class extracts numerical gradient components from a text file and
    reconstructs the Cartesian gradient vector corresponding to a molecular
    geometry. The reader is intended for use in finite-difference Hessian
    calculations, where gradients computed by an electronic structure
    program are used to construct force-constant matrices.

    All floating-point values are extracted from the file using a regular
    expression. The Cartesian gradient is assumed to be stored as the final
    ``3N`` values in the file, where ``N`` is the number of atoms in the
    molecular geometry.

    Parameters
    ----------
    file_name : str
        Name of the gradient file to be read.

    Attributes
    ----------
    file_name : str
        Input filename.

    grad_regex : re.Pattern
        Regular expression used to extract floating-point values from the
        gradient file.

    grad : list[str]
        Raw numerical values extracted from the file.

    cart_grad : ndarray
        Cartesian gradient vector containing the final ``3N`` gradient
        components.

    Notes
    -----
    The reader assumes that the gradient information appears at the end of
    the file and that the total number of Cartesian components is equal to

    .. math::

        3N

    where ``N`` is the number of atoms.

    This utility is primarily used during gradient-based finite-difference
    Hessian construction within the Concordant Modes Algorithm (CMA)
    workflow.

    Examples
    --------
    >>> reader = GrRead("fc_b.grad")
    >>> reader.run(cartesians)
    >>> grad = reader.cart_grad
    >>> grad.shape
    (3 * len(cartesians),)
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.grad_regex = re.compile(r"(-?\d+\.\d+)")

    def run(self, cartesians):
        """
        Read and extract the Cartesian gradient vector.

        Parameters
        ----------
        cartesians : ndarray
            Cartesian coordinates of the molecular geometry. The number of
            coordinates is used to determine the expected length of the
            Cartesian gradient vector.

        Returns
        -------
        None

        Notes
        -----
        All floating-point values are extracted from the file, and the final
        ``3N`` values are interpreted as the Cartesian gradient components,
        where ``N`` is the number of atoms.

        The resulting gradient vector is stored in the attribute
        ``cart_grad``.
        """
        with open(self.file_name, "r") as file:
            grad = file.read()
        self.grad = re.findall(self.grad_regex, grad)
        self.cart_grad = np.array(self.grad).astype(float)[-len(cartesians.flatten()) :]
