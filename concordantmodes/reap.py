import numpy as np
import json
import os
import shutil
import re

from concordantmodes.s_vectors import SVectors


class Reap(object):
    def __init__(
        self,
        options,
        num_deg_free,
        indices,
        symm_obj,
        cma_level,
        deriv_level=0,
        disp_sym=None,
        disp=None,
        ted=None,
        zmat=None
    ):
        self.options = options
        self.num_deg_free = num_deg_free
        self.indices = indices
        self.symm_obj = symm_obj
        self.deriv_level = deriv_level
        self.cma_level = cma_level
        self.disp = disp
        self.ted = ted
        self.zmat = zmat
        if cma_level == "B":
            if self.deriv_level:
                self.gradient_regex = self.options.gradient_regex_b
                self.energy_regex = self.options.energy_regex_b
                self.success_regex = self.options.success_regex_b
            else:
                self.energy_regex = self.options.energy_regex_b
                self.success_regex = self.options.success_regex_b
        elif cma_level == "C":
            if self.deriv_level:
                pass
            else:
                self.energy_regex = self.options.energy_regex_c
                self.success_regex = self.options.success_regex_c
        else:  # cma_level = "A"
            if self.deriv_level:
                self.gradient_regex = self.options.gradient_regex_a
                self.energy_regex = self.options.energy_regex_a
                self.success_regex = self.options.success_regex_a
            else:
                self.energy_regex = self.options.energy_regex_a
                self.success_regex = self.options.success_regex_a

    def run(self):
        # Define energy/gradient search regex
        if not self.deriv_level:
            energy_regex = re.compile(self.energy_regex)
            success_regex = re.compile(self.success_regex)
            self.energies = np.array([])
        else:
            grad_regex1 = re.compile(self.gradient_regex[0])
            grad_regex2 = re.compile(self.gradient_regex[1])
        size = self.num_deg_free

        # if self.options.second_order:
        # print(self.indices)
        # print(self.indices[-1])
        # raise RuntimeError
        # size = self.indices[-1][0] + 1

        self.fail_list = []
        if not self.deriv_level:
            print(
                "If something looks wrong with the final frequencies, check these energies!"
            )
            print("(Job number 1 == Reference energy) :D")
            print(os.getcwd())
            os.chdir("./" + str(1))
            with open("output.dat", "r") as file:
                data = file.read()
            print(f"Success regex {success_regex}")
            if not re.search(success_regex, data):
                print("Energy failed at " + str("ref"))
                raise RuntimeError
            os.chdir("..")

            ref_en = float(re.findall(energy_regex, data)[0])
            print("Reference energy: " + str(ref_en))

            indices = self.indices
            p_en_array = np.zeros((size, size))
            m_en_array = np.zeros((size, size))
            rel_en_p = np.zeros((size, size))
            rel_en_m = np.zeros((size, size))
            relative_energies = []
            absolute_energies = [[("ref", "ref"), "ref", ref_en, 1]]

            direc = 2

            if self.symm_obj.symtext is not None and self.options.exploit_pm_symm:
                if self.options.only_TSIR:
                    print("Reap only the TSIR displacements")
                    indices = self.symm_obj.indices_by_irrep[0]
                else:
                    print("Reap displacements from all irreps")
                    indices = self.symm_obj.indices_by_irrep.flatten().reshape((-1, 2))

            Sum = 1
            h = 0
            # print(p_en_array.shape)
            # print(p_en_array)
            # print(indices)
            for index in indices:
                i, j = index[0], index[1]
                p_en_array[i, j] = energy = self.reap_energies(
                    direc, success_regex, energy_regex
                )
                print("p_en")
                print(energy)
                rel = energy - ref_en
                print(
                    "Relative plus  "
                    + "{:4d}".format(direc)
                    + "{:4d}".format(i)
                    + " "
                    + "{:4d}".format(j)
                    + ": "
                    + "{: 10.9f}".format(rel)
                )
                rel_en_p[i, j] = rel
                relative_energies.append([(i, j), "plus", rel, direc])
                absolute_energies.append([(i, j), "plus", energy, direc])

                if h != 0:
                    m_en_array[i, j] = energy = self.reap_energies(
                        direc, success_regex, energy_regex
                    )
                    direc += 1
                else:
                    m_en_array[i, j] = energy = self.reap_energies(
                        direc + 1, success_regex, energy_regex
                    )
                    print("m_en")
                    print(energy)
                    rel = energy - ref_en
                    print(
                        "Relative minus "
                        + "{:4d}".format(direc + 1)
                        + "{:4d}".format(i)
                        + " "
                        + "{:4d}".format(j)
                        + ": "
                        + "{: 10.9f}".format(rel)
                    )
                    rel_en_m[i, j] = rel
                    relative_energies.append([(i, j), "minus", rel, direc + 1])
                    absolute_energies.append([(i, j), "minus", energy, direc + 1])
                    direc += 2

                if (
                    self.symm_obj.symtext is not None
                    and self.options.exploit_pm_symm
                    and not self.options.only_TSIR
                    and Sum > len(self.symm_obj.indices_by_irrep[0])
                ):
                    h += 1
                Sum += 1

            self.p_en_array = p_en_array
            self.m_en_array = m_en_array
            self.ref_en = ref_en
            print_en = absolute_energies
            np.set_printoptions(precision=2, linewidth=120)
            os.chdir("..")

        else:
            indices = self.indices
            p_grad_array = np.array([], dtype=object)
            m_grad_array = np.array([], dtype=object)
            Sum = 0
            print(indices)
            for index in indices:
                grad = self.reap_gradients(
                    2 * index[0] + 1 - Sum, grad_regex1, grad_regex2
                )
                p_grad_array = np.append(p_grad_array, grad, axis=0)
                grad = self.reap_gradients(
                    2 * index[0] + 2 - Sum, grad_regex1, grad_regex2
                )
                m_grad_array = np.append(m_grad_array, grad, axis=0)
            self.p_grad_array = p_grad_array.reshape((-1, len(grad)))
            self.m_grad_array = m_grad_array.reshape((-1, len(grad)))
            
            zmat_bool = self.zmat is not None
            ted_bool = self.ted is not None
            disp_bool = self.disp is not None

            if self.options.conv_grad and zmat_bool and ted_bool and disp_bool:
                grad_s_vec = SVectors(
                    self.zmat,
                    self.options,
                )

                for i in range(len(indices)):
                    grad_s_vec.run(disp.p_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.ted.proj)
                    self.p_array_b[i] = np.dot(cart_p_array_b[i].T, A_proj)

                    grad_s_vec.run(disp.m_disp[i], False)
                    A_proj = np.dot(LA.pinv(grad_s_vec.B), self.ted.proj)
                    self.m_array_b[i] = np.dot(cart_m_array_b[i].T, A_proj)
                
            os.chdir("..")
        if len(self.fail_list):
            print("Some jobs have failed:")
            print(self.fail_list)
            raise RuntimeError

    def reap_energies(self, direc, success_regex, energy_regex):
        os.chdir("./" + str(direc))

        with open("output.dat", "r") as file:
            data = file.read()

        if not re.search(success_regex, data):
            self.fail_list.append(direc)
            energy = 0.0
        else:
            energy = float(re.findall(energy_regex, data)[0])

        os.chdir("..")

        return energy

    def reap_gradients(self, direc, grad_regex1, grad_regex2):
        os.chdir("./" + str(direc))
        grad_array = []
        if self.options.program_b == "molpro":
            with open("output.xml", "r") as file:
                data = file.readlines()
        else:
            with open("output.dat", "r") as file:
                data = file.readlines()
        for i in range(len(data)):
            grad1 = re.search(grad_regex1, data[i])
            if grad1:
                beg_grad = i + 1
                break
        for i in range(len(data) - beg_grad):
            grad2 = re.search(grad_regex2, data[i + beg_grad])
            if grad2:
                end_grad = i + beg_grad
                break
        label_xyz = r"(\s*.*(\s*-?\d+\.\d+){3})+"
        for line in data[beg_grad:end_grad]:
            if re.search(label_xyz, line):
                temp = line.split()[-3:]
                grad_array.append(temp)
        grad_array = np.array(grad_array)
        grad_array = grad_array.astype("float64")
        grad_array = grad_array.flatten()
        if not grad1:
            print("Gradient failed at " + os.getcwd())
            raise RuntimeError
        os.chdir("..")

        return grad_array
