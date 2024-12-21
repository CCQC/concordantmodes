import numpy as np
from copy import copy
import scipy.linalg as la
import time


class RMSD(object):
    def __init__(self):
        print("Nothing to init")

    # computes the moment of inertia given a molecule and the axis

    def compute_IAB(self, mol1, mol2, n):
        iab = np.zeros((3, 3))
        for a in range(0, 3):
            for b in range(0, 3):
                if a == b:
                    kron = 1
                else:
                    kron = 0
                for k in range(0, n):
                    iab[a, b] += (
                        mol1[k, 0] * mol2[k, 0]
                        + mol1[k, 1] * mol2[k, 1]
                        + mol1[k, 2] * mol2[k, 2]
                    ) * kron - mol1[k, a] * mol2[k, b]
        return iab

    # locate centroid of molecule

    def centroid(self, mol, n):
        cent = 0
        for i, x in enumerate(mol):
            cent += x
        cent = cent / n
        return cent

    # translate centroid of molecule to origin

    def translate(self, mol, n):
        molt = np.zeros((n, 3))
        mol_cent = self.centroid(mol, n)
        for i, x in enumerate(mol):
            molt[i] = x - mol_cent
        return molt

    # compute mixed inertia tensor

    def compute_QAB(self, iab):
        qab = np.array(
            [
                [iab[0, 0], iab[0, 1], iab[0, 2], iab[1, 2]],
                [iab[1, 0], iab[1, 1], iab[1, 2], iab[2, 0]],
                [iab[2, 0], iab[2, 1], iab[2, 2], iab[0, 1]],
                [-iab[2, 1], -iab[0, 2], -iab[1, 0], 0],
            ]
        )
        return qab

    # the whole point of this code, minimize this

    def compute_SminX(self, mol1, mol2, qab, n):
        SminX = 0
        for k in range(0, n):
            SminX += (np.linalg.norm(mol1[k] - mol2[k])) ** 2
            w, v = np.linalg.eig(qab + qab.T)
            ind = np.argmin(w)
        SminX += 2 * w[ind]
        return SminX

    # compute moments of inertia

    def compute_moi(self, mol, n):
        mxx, myy, mzz = 0.0, 0.0, 0.0
        for k in range(0, n):
            mxx += mol[k, 0] ** 2
            myy += mol[k, 1] ** 2
            mzz += mol[k, 2] ** 2
        return mxx, myy, mzz

    # compute products of intertia

    def compute_poi(self, mol, n):
        mxy, mxz, myz = 0.0, 0.0, 0.0
        for k in range(0, n):
            mxy += mol[k, 0] * mol[k, 1]
            mxz += mol[k, 0] * mol[k, 2]
            myz += mol[k, 1] * mol[k, 2]
        return mxy, mxz, myz

    # check signs of x, y, and z coordinates

    def sign_check(self, mol1_rot, mol2_rot):
        s12 = [0.0, 0.0, 0.0]
        s1 = (np.sign(mol1_rot)).T
        s2 = (np.sign(mol2_rot)).T
        for k in range(0, 3):
            s12[k] = np.sign(np.sum(s1[k] * s2[k]))
        return s12

    # change signs of x, y, and z coordinates using s12 vector values

    def sign_change(self, mol1_rot, mol2_rot, s12, n):
        for j in range(0, 3):
            if s12[j] != 0:
                for k in range(0, n):
                    mol2_rot[k, j] = s12[j] * mol2_rot[k, j]
        return mol2_rot

    # compute resultant vector between two sets of cartesians

    def compute_res0(self, mol1_rot, mol2_rot, n):
        res0 = []
        for k in range(0, n):
            result = mol1_rot[k] - mol2_rot[k]
            result.flatten()
            res0.append(result)
        res0 = np.array(res0)
        res0 = res0.flatten()
        return res0

    # compute rmsd using resultant vector

    def compute_rmsd(self, res, n):
        rmsd = np.sqrt(np.dot(res, res) / n)
        return rmsd

    def mol_test(self, mol, n):
        check = 0
        for k in range(0, n):
            check += mol[k]
        return check

    # First and Second derivatives of the Euler-Rodrigues rotation matrix derived and "simplified" in mathematica.

    def norm(self, grud):
        return np.sqrt(abs(grud[0]) ** 2 + abs(grud[1]) ** 2 + abs(grud[2]) ** 2)

    def handwritten_hess(self, mol1, mol2, n):
        A = 0
        B = 0
        G = 0
        for z in range(0, 5):
            grad0, grad1, grad2, hess00, hess01, hess02, hess11, hess12, hess22 = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            for k in range(0, n):
                c1 = mol2[k, 0]
                c2 = mol2[k, 1]
                c3 = mol2[k, 2]
                K1 = mol1[k, 0]
                K2 = mol1[k, 1]
                K3 = mol1[k, 2]
                grad0 += (
                    4
                    * (
                        -c1
                        * (
                            -K3 * A * B
                            + K2 * A * G
                            + K2 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K3 * G * np.sqrt(1 - A**2 - B**2 - G**2)
                        )
                        + c2
                        * (
                            K1 * A * G
                            + 2 * K2 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            - K1 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                            - K3 * (-1 + 2 * A**2 + B**2 + G**2)
                        )
                        + c3
                        * (
                            2 * K3 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K2 * (-1 + 2 * A**2 + B**2 + G**2)
                            - K1 * (A * B + G * np.sqrt(1 - A**2 - B**2 - G**2))
                        )
                    )
                ) / np.sqrt(1 - A**2 - B**2 - G**2)

                grad1 += -(
                    4
                    * (
                        c2
                        * (
                            K3 * A * B
                            - K1 * B * G
                            + K1 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K3 * G * np.sqrt(1 - A**2 - B**2 - G**2)
                        )
                        + c3
                        * (
                            -K2 * A * B
                            - 2 * K3 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K2 * G * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K1 * (-1 + 2 * A**2 + B**2 + G**2)
                        )
                        + c1
                        * (
                            K2 * B * G
                            + K2 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            - 2 * K1 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                            - K3 * (-1 + A**2 - 2 * B**2 + G**2)
                        )
                    )
                ) / np.sqrt(1 - A**2 - B**2 - G**2)

                grad2 += -(
                    4
                    * (
                        c3
                        * (
                            -K2 * A * G
                            + K1 * B * G
                            + K1 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K2 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                        )
                        + c2
                        * (
                            K3 * A * G
                            + K3 * B * np.sqrt(1 - A**2 - B**2 - G**2)
                            - 2 * K2 * G * np.sqrt(1 - A**2 - B**2 - G**2)
                            - K1 * (-1 + A**2 + B**2 + 2 * G**2)
                        )
                        + c1
                        * (
                            -K3 * B * G
                            + K3 * A * np.sqrt(1 - A**2 - B**2 - G**2)
                            - 2 * K1 * G * np.sqrt(1 - A**2 - B**2 - G**2)
                            + K2 * (-1 + A**2 + B**2 + 2 * G**2)
                        )
                    )
                ) / np.sqrt(1 - A**2 - B**2 - G**2)

                hess00 += -(
                    (
                        4
                        * (
                            c1 * (K3 * B - K2 * G) * (-1 + B**2 + G**2)
                            + c3
                            * (
                                2 * K2 * A**3
                                - 2 * K3 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                                + 3 * K2 * A * (-1 + B**2 + G**2)
                                - K1 * B * (-1 + B**2 + G**2)
                            )
                            + c2
                            * (
                                -2 * K3 * A**3
                                - 2 * K2 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                                - 3 * K3 * A * (-1 + B**2 + G**2)
                                + K1 * G * (-1 + B**2 + G**2)
                            )
                        )
                    )
                    / (1 - A**2 - B**2 - G**2) ** (3 / 2)
                )

                hess01 += (
                    4
                    * (
                        -c1 * K3 * A * (-1 + A**2 + G**2)
                        + c2 * K3 * B * (-1 + B**2 + G**2)
                        + c3
                        * (
                            K1 * A * (-1 + A**2 + G**2)
                            - K2 * B * (-1 + B**2 + G**2)
                        )
                        + c1
                        * K2
                        * (
                            -A * B * G
                            + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                            + np.sqrt(1 - A**2 - B**2 - G**2)
                            * (-1 + B**2 + G**2)
                        )
                        + c2
                        * K1
                        * (
                            A * B * G
                            + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                            + np.sqrt(1 - A**2 - B**2 - G**2)
                            * (-1 + B**2 + G**2)
                        )
                    )
                ) / (1 - A**2 - B**2 - G**2) ** (3 / 2)

                hess02 += (
                    4
                    * (
                        c1 * K2 * A * (-1 + A**2 + B**2)
                        - c3 * K2 * G * (-1 + B**2 + G**2)
                        + c2
                        * (
                            -K1 * A * (-1 + A**2 + B**2)
                            - K3 * G * (-1 + B**2 + G**2)
                        )
                        + c3
                        * K1
                        * (
                            -A * B * G
                            + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                            + np.sqrt(1 - A**2 - B**2 - G**2)
                            * (-1 + B**2 + G**2)
                        )
                        + c1
                        * K3
                        * (
                            A * B * G
                            + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                            + np.sqrt(1 - A**2 - B**2 - G**2)
                            * (-1 + B**2 + G**2)
                        )
                    )
                ) / (1 - A**2 - B**2 - G**2) ** (3 / 2)

                hess11 += -(
                    (
                        4
                        * (
                            -c2 * (K3 * A - K1 * G) * (-1 + A**2 + G**2)
                            + c3
                            * (
                                -2 * K3 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                                + K2 * A * (-1 + A**2 + G**2)
                                - K1 * B * (-3 + 3 * A**2 + 2 * B**2 + 3 * G**2)
                            )
                            + c1
                            * (
                                -2 * K1 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                                - K2 * G * (-1 + A**2 + G**2)
                                + K3 * B * (-3 + 3 * A**2 + 2 * B**2 + 3 * G**2)
                            )
                        )
                    )
                    / (1 - A**2 - B**2 - G**2) ** (3 / 2)
                )

                hess12 += (
                    4
                    * (
                        c1 * K2 * B * (-1 + A**2 + B**2)
                        + c3 * K1 * G * (-1 + A**2 + G**2)
                        - c1 * K3 * G * (-1 + A**2 + G**2)
                        + c3
                        * K2
                        * (
                            A * B * G
                            + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                            + np.sqrt(1 - A**2 - B**2 - G**2)
                            * (-1 + B**2 + G**2)
                        )
                        + c2
                        * (
                            -K1 * B * (-1 + A**2 + B**2)
                            + K3
                            * (
                                -A * B * G
                                + (A**2) * np.sqrt(1 - A**2 - B**2 - G**2)
                                + np.sqrt(1 - A**2 - B**2 - G**2)
                                * (-1 + B**2 + G**2)
                            )
                        )
                    )
                ) / (1 - A**2 - B**2 - G**2) ** (3 / 2)

                hess22 += -(
                    (
                        4
                        * (
                            c3 * (K2 * A - K1 * B) * (-1 + A**2 + B**2)
                            + c1
                            * (
                                K3 * B * (-1 + A**2 + B**2)
                                - 3 * K2 * (-1 + A**2 + B**2) * G
                                - 2 * K2 * G**3
                                - 2 * K1 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                            )
                            + c2
                            * (
                                -K3 * A * (-1 + A**2 + B**2)
                                + 3 * K1 * (-1 + A**2 + B**2) * G
                                + 2 * K1 * G**3
                                - 2 * K2 * (1 - A**2 - B**2 - G**2) ** (3 / 2)
                            )
                        )
                    )
                    / (1 - A**2 - B**2 - G**2) ** (3 / 2)
                )
            grud = np.array([grad0, grad1, grad2])
            huss = np.array(
                [
                    [hess00, hess01, hess02],
                    [hess01, hess11, hess12],
                    [hess02, hess12, hess22],
                ]
            )
            bars = np.dot(grud, np.linalg.inv(huss))
            if self.norm(grud) < 1e-12:
                print("Optimization of Euler-angles optimization has converged")
                print(f"Norm of gradient is {self.norm(grud)}")
                break
            A = A - bars[0]
            B = B - bars[1]
            G = G - bars[2]

        return A, B, G

    def derive_opt_rotate(self, mol1_rot, mol2_rot, n):  # , oilrod):
        a, b, c = self.handwritten_hess(mol1_rot, mol2_rot, n)
        print(a, b, c)

        rot_matrix = np.array(
            [
                [
                    1 - 2 * b**2 - 2 * c**2,
                    2 * (a * b + c * np.sqrt(1 - a**2 - b**2 - c**2)),
                    2 * (a * c - b * np.sqrt(1 - a**2 - b**2 - c**2)),
                ],
                [
                    2 * (a * b - c * np.sqrt(1 - a**2 - b**2 - c**2)),
                    1 - 2 * a**2 - 2 * c**2,
                    2 * (b * c + a * np.sqrt(1 - a**2 - b**2 - c**2)),
                ],
                [
                    2 * (a * c + b * np.sqrt(1 - a**2 - b**2 - c**2)),
                    2 * (b * c - a * np.sqrt(1 - a**2 - b**2 - c**2)),
                    1 - 2 * a**2 - 2 * b**2,
                ],
            ]
        )

        mol2Final = (np.dot(rot_matrix, mol2_rot.T)).T
        return mol2Final

    def eigen_convention(self, v1, v2):
        for i in range(0, v2.shape[0]):
            rowmax = np.argmax(abs(v2[i]))
            if v2[i, rowmax] < 0:
                v2[i] = v2[i] * -1
                v1[i] = v1[i] * -1
        return v1, v2

    def run(self, mol1, mol2):
        start = time.time()
        n = mol2.shape[0]

        mol1 = self.translate(mol1, n)
        mol2 = self.translate(mol2, n)

        iab = self.compute_IAB(mol1, mol2, n)
        qab = self.compute_QAB(iab)

        SminX = self.compute_SminX(mol1, mol2, qab, n)
        mxx1, myy1, mzz1 = self.compute_moi(mol1, n)
        mxx2, myy2, mzz2 = self.compute_moi(mol2, n)
        mxy1, mxz1, myz1 = self.compute_poi(mol1, n)
        mxy2, mxz2, myz2 = self.compute_poi(mol2, n)

        M1_I = np.array(
            [
                [myy1 + mzz1, -mxy1, -mxz1],
                [-mxy1, mxx1 + mzz1, -myz1],
                [-mxz1, -myz1, mxx1 + myy1],
            ]
        )

        M2_I = np.array(
            [
                [myy2 + mzz2, -mxy2, -mxz2],
                [-mxy2, mxx2 + mzz2, -myz2],
                [-mxz2, -myz2, mxx2 + myy2],
            ]
        )

        W1, V1 = np.linalg.eig(M1_I)
        W2, V2 = np.linalg.eig(M2_I)
        # sort eigensolutions, not sorted because np only autosorts eigenvalues of self-adjoint systems using np.eigh
        idx1 = W1.argsort()[::-1]
        W1 = W1[idx1]
        V1 = V1[:, idx1]

        idx2 = W2.argsort()[::-1]
        W2 = W2[idx2]
        V2 = V2[:, idx2]

        V1 = V1.T
        V2 = V2.T
        # for troubleshooting purposes, this function changed the eigenvector sign convention to match mathematica's
        # it will not affect the final rmsd value, just the sign convention on the molecule
        # V1, V2 = self.eigen_convention(V1, V2)

        mol1_rot = (np.dot(V1, mol1.T)).T
        mol2_rot = (np.dot(V2, mol2.T)).T

        s12 = self.sign_check(mol1_rot, mol2_rot)

        mol2_rot = self.sign_change(mol1_rot, mol2_rot, s12, n)

        iab_rot = self.compute_IAB(mol1_rot, mol2_rot, n)
        qab_rot = self.compute_QAB(iab_rot)
        SminX_rot = self.compute_SminX(mol1_rot, mol2_rot, qab_rot, n)

        res0 = self.compute_res0(mol1_rot, mol2_rot, n)
        rmsd0 = self.compute_rmsd(res0, n)

        print(f"RMSD prior to optimization of the rotatation parameters {rmsd0}")
        mxx1, myy1, mzz1 = self.compute_moi(mol1_rot, n)
        mxx2, myy2, mzz2 = self.compute_moi(mol2_rot, n)
        mxy1, mxz1, myz1 = self.compute_poi(mol1_rot, n)
        mxy2, mxz2, myz2 = self.compute_poi(mol2_rot, n)

        M1_I = np.array(
            [
                [myy1 + mzz1, -mxy1, -mxz1],
                [-mxy1, mxx1 + mzz1, -myz1],
                [-mxz1, -myz1, mxx1 + myy1],
            ]
        )

        M2_I = np.array(
            [
                [myy2 + mzz2, -mxy2, -mxz2],
                [-mxy2, mxx2 + mzz2, -myz2],
                [-mxz2, -myz2, mxx2 + myy2],
            ]
        )

        print("CHECK: Is molecule at origin? These should be zero vectors")
        print(self.mol_test(mol1_rot, n))
        print(
            """     

        """
        )
        print(self.mol_test(mol2_rot, n))
        print(
            """     

        """
        )
        print("CHECK: Are the products of inertia zero?")
        print(M1_I)
        print(
            """     

        """
        )
        print(M2_I)
        print(
            """     

        """
        )
        print("Beginning optimization of Euler-angle parameters")
        mol2_rot = self.derive_opt_rotate(mol1_rot, mol2_rot, n)

        resfinal = self.compute_res0(mol1_rot, mol2_rot, n)
        rmsdfinal = self.compute_rmsd(resfinal, n)
        print("rmsdfinal")
        print(rmsdfinal)
        end = time.time()
        print(f"It took {end - start} seconds to compute the rmsd of {rmsdfinal}")
        self.rmsd = rmsdfinal
