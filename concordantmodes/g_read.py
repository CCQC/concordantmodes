import numpy as np
import re
from numpy.linalg import inv
from numpy import linalg as LA


class GrRead(object):
    """
    This class may be used to read in force constant information from the
    standard 3-column formatted file. Idk what that format is called, but
    it's pretty standard.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.grad_regex = re.compile(r"(-?\d+\.\d+)")

    def run(self, cartesians):
        with open(self.file_name, "r") as file:
            grad = file.read()
        self.grad = re.findall(self.grad_regex, grad)
        self.cart_grad = np.array(self.grad).astype(float)[-len(cartesians.flatten()) :]
