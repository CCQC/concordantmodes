import numpy as np
from numpy import linalg as LA
from . import masses


class Int2Cart(object):
    def __init__(self, zmat, var_dict=None):
        self.zmat = zmat
        self.var_dict = var_dict
        self.masses = [masses.get_mass(label) for label in zmat.atom_list]
        self.tol = 1e-14

    def run(self):
        # First atom is placed at the origin.
        self.carts = np.array([[0.0, 0.0, 0.0]])

        # Second atom is placed along the x-axis.
        self.carts = np.append(
            self.carts,
            np.array([[self.var_dict[self.zmat.bond_variables[0]], 0.0, 0.0]]),
            axis=0,
        )

        # Third atom is rotated from the x-axis into the xy-plane depending upon the angle btw atoms 1, 2, and 3.
        if len(self.zmat.atom_list) > 2:
            r = (
                self.carts[int(self.zmat.angle_indices[0][2]) - 1]
                - self.carts[int(self.zmat.angle_indices[0][1]) - 1]
            )
            r /= np.linalg.norm(r)
            r *= self.var_dict[self.zmat.bond_variables[1]]
            theta = self.var_dict[self.zmat.angle_variables[0]] * np.pi / 180.0
            ang_rot = self.x_rot(theta)
            cart = (
                np.dot(r, ang_rot) + self.carts[int(self.zmat.angle_indices[0][1]) - 1]
            )
            self.carts = np.append(self.carts, np.array([cart]), axis=0)

        # Now for the fun part, generalization for the torsions!
        if len(self.zmat.atom_list) > 2:
            for i in range(len(self.zmat.torsion_variables)):
                # This code will rotate a unit vector along the x-axis to the proper orientation, as if it were along the prior bond.
                # Then the vector will be rotated the proper angles to move the x-axis unit vector in line with the prior bond unit vector.
                # Finally, the vector will be shifted by the cartesian cordinates of the atom that our new atom is bonded to.
                xUnit = np.array([1, 0, 0])
                r = (
                    self.carts[int(self.zmat.torsionIndices[i][2]) - 1]
                    - self.carts[int(self.zmat.torsionIndices[i][1]) - 1]
                )
                r /= np.linalg.norm(r)
                yRot = np.array([r[0].copy(), 0, r[2].copy()])
                yRot /= LA.norm(yRot)
                yRotMat = np.eye(3)
                yRotMat[0][0] = yRot[0].copy()
                yRotMat[2][2] = yRot[0].copy()
                yRotMat[0][2] = yRot[2].copy()
                yRotMat[2][0] = -yRot[2].copy()
                zRot = np.array([r[0].copy(), r[1].copy(), 0])
                zRot /= LA.norm(zRot)
                zRotMat = np.eye(3)
                zRotMat[0][0] = zRot[0].copy()
                zRotMat[1][1] = zRot[0].copy()
                zRotMat[0][1] = zRot[1].copy()
                zRotMat[1][0] = -zRot[1].copy()
                theta = self.var_dict[self.zmat.angle_variables[i + 1]] * np.pi / 180.0
                tau = self.var__dict[self.zmat.torsion_variables[i]] * np.pi / 180.0
                ang_rot = self.x_rot(theta)
                tor_rot = self.z_rot(tau)
                cart = np.dot(np.dot(r, ang_rot), tor_rot)
                cart = np.dot(np.dot(cart, yRotMat), zRotMat)
                cart *= self.var_dict[self.zmat.bond_variables[i + 2]]
                cart += self.carts[int(self.zmat.torsionIndices[i][1]) - 1]
                self.carts = np.append(self.carts, np.array([cart]), axis=0)

        C_O_M = self.COM(self.carts, self.zmat.masses)
        for i in range(len(self.carts)):
            self.carts[i] -= C_O_M
        C_O_M = self.COM(self.carts, self.zmat.masses)
        I = self.InertiaTensor(self.carts, self.zmat.masses)
        val, vec = LA.eigh(I)
        self.carts = np.dot(self.carts, vec)
        self.carts[np.abs(self.carts) < self.tol] = 0.0

    def InertiaTensor(self, carts, masses):
        mass_weighted = np.multiply(np.sqrt(masses), carts.transpose())
        In_Tens = np.dot(mass_weighted, mass_weighted.transpose())
        I = np.eye(3)
        I *= np.trace(In_Tens)
        I -= In_Tens
        return I

    def COM(self, carts, masses):
        mass_weighted = np.dot(masses, carts)
        com = mass_weighted / np.sum(masses)
        return com

    def x_rot(self, theta):
        x_rot_mat = np.identity(3)
        x_rot_mat[0, 0] = np.cos(theta)
        x_rot_mat[1, 1] = np.cos(theta)
        x_rot_mat[1, 0] = -np.sin(theta)
        x_rot_mat[0, 1] = np.sin(theta)
        return x_rot_mat

    def z_rot(self, tau):
        z_rot_mat = np.identity(3)
        z_rot_mat[1, 1] = np.cos(tau)
        z_rot_mat[2, 2] = np.cos(tau)
        z_rot_mat[2, 1] = -np.sin(tau)
        z_rot_mat[1, 2] = np.sin(tau)
        return z_rot_mat
