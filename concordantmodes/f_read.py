import numpy as np
import re
from numpy.linalg import inv
from numpy import linalg as LA


class FcRead(object):
    """
    This class may be used to read in force constant information from the
    standard 3-column formatted file. Idk what that format is called, but
    it's pretty standard.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.fc_regex = re.compile(r"(-?\d+\.\d+)")

    def run(self):
        with open(self.file_name, "r") as file:
            fc = file.read()

        self.fc_mat = re.findall(self.fc_regex, fc)
        self.fc_mat = np.reshape(
            self.fc_mat, (int(np.sqrt(len(self.fc_mat))), -1)
        ).astype(float)
