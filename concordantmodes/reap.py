import numpy as np
import json
import os
import shutil
import re


class Reap(object):
    def __init__(
        self,
        prog_name,
        zmat,
        disp_cart,
        options,
        n_coord,
        eigs,
        indices,
        energy_regex,
        gradient_regex,
        molly_regex,
        success_regex,
        deriv_level=0,
        disp_sym=None,
    ):
        self.prog_name = prog_name
        self.zmat = zmat
        self.disp_cart = disp_cart
        self.options = options
        self.n_coord = n_coord
        # self.disp_sym =disp_sym
        # nate
        self.eigs = eigs
        self.energy_regex = energy_regex
        self.gradient_regex = gradient_regex
        self.molly_regex = molly_regex
        self.success_regex = success_regex
        self.indices = indices
        self.deriv_level = deriv_level

    def run(self):
        # Define energy/gradient search regex
        if not self.deriv_level:
            energy_regex = re.compile(self.energy_regex)
            success_regex = re.compile(self.success_regex)
            self.energies = np.array([])
        else:
            grad_regex1 = re.compile(self.gradient_regex[0])
            grad_regex2 = re.compile(self.gradient_regex[1])
            # molly_regex1  = re.compile(self.molly_regex[0])
            # molly_regex2  = re.compile(self.molly_regex[1])
        eigs = self.eigs
        if type(eigs) == int:
            size = eigs
        else:
            size = len(eigs)
        # n_disp = len(self.disp_cart)

        if not self.deriv_level:
            print(
                "If something looks wrong with the final frequencies, check these energies!"
            )
            print("(Job number 1 == Reference energy) :D")
            print(os.getcwd())
            # self.options.dir_reap = True
            if self.options.dir_reap:
                os.chdir("./" + str(1))
                with open("output.dat", "r") as file:
                    data = file.read()
                if not re.search(success_regex, data):
                    print("Energy failed at " + str("ref"))
                    raise RuntimeError
                os.chdir("..")
            else:
                with open("output.1.dat", "r") as file:
                    data = file.read()
                if not re.search(success_regex, data):
                    print("Energy failed at " + str("ref"))
                    raise RuntimeError

            ref_en = float(re.findall(energy_regex, data)[0])
            print("Reference energy: " + str(ref_en))

            indices = self.indices
            # eigs = self.eigs
            # eigs = len(eigs)
            p_en_array = np.zeros((size, size))
            m_en_array = np.zeros((size, size))
            rel_en_p = np.zeros((size, size))
            rel_en_m = np.zeros((size, size))
            relative_energies = []
            absolute_energies = [[("ref", "ref"), "ref", ref_en, 1]]

            direc = 2
            for index in indices:
                i, j = index[0], index[1]
                p_en_array[i, j] = energy = self.reap_energies(
                    direc, success_regex, energy_regex
                )
                # print(energy)
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

                # if self.disp_sym[i] and self.disp_sym[j]:
                # direc -= 1
                # m_en_array[i, j] = energy = p_en_array[i, j]
                # else:
                # m_en_array[i, j] = energy = self.reap_energies(
                # direc + 1, success_regex, energy_regex
                # )
                # print(energy)
                m_en_array[i, j] = energy = self.reap_energies(
                    direc + 1, success_regex, energy_regex
                )
                # print(energy)
                rel = energy - ref_en
                # print("Relative minus " + str(i) + " " + str(j) + ": " + "{:10.6f}".format(rel))
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

            self.p_en_array = p_en_array
            self.m_en_array = m_en_array
            self.ref_en = ref_en
            # print_en = np.insert(absolute_energies,0,[("ref", "ref"), "ref", ref_en, 1],axis=0)
            print_en = absolute_energies
            np.set_printoptions(precision=2, linewidth=120)
            print(
                "Relative energies plus-displacements on the diagonal and plus/plus-displacements on the off-diagonal elements"
            )
            print(rel_en_p)
            print(
                "Relative energies minus-displacements on the diagonal and minus/minus-displacements on the off-diagonal elements"
            )
            print(rel_en_m)
            os.chdir("..")
            # if self.options.printout_rel_e:
            # auxiliary = ""
            # header = "Index, relative energy, directory \n"
            # print(json.dumps(energy))
            # with open("auxiliary", "a") as file:
            # file.seek(0)
            # file.truncate()
            # file.writelines(header)
            # for energy in print_en:
            # with open("auxiliary", "a") as file:
            # file.writelines(str(energy) + "\n")
        else:
            indices = self.indices
            p_grad_array = np.array([])
            m_grad_array = np.array([])
            Sum = 0
            # print(indices)
            for index in indices:
                grad = self.reap_gradients(
                    2 * index + 1 - Sum, grad_regex1, grad_regex2
                )
                p_grad_array = np.append(p_grad_array, grad, axis=0)
                # if self.disp.disp_sym[i]:
                # m_grad_array = np.append(m_grad_array, -grad, axis=0)
                # Sum += 1
                # else:
                # grad = self.reap_gradients(2 * index + 2 - Sum, grad_regex1, grad_regex2)
                # m_grad_array = np.append(m_grad_array, grad, axis=0)
                grad = self.reap_gradients(
                    2 * index + 2 - Sum, grad_regex1, grad_regex2
                )
                m_grad_array = np.append(m_grad_array, grad, axis=0)
            self.p_grad_array = p_grad_array.reshape((-1, len(grad)))
            self.m_grad_array = m_grad_array.reshape((-1, len(grad)))
            os.chdir("..")

    def reap_molly(self, direc, molly1_regex, molly2_regex):
        os.chdir("./" + str(direc))
        with open("output.dat", "r") as file:
            datta = file.read()
        # try to grab init geom
        initmolreg = r"\s*\{([^}]+)\}"
        reggie = re.compile(initmolreg)
        initmol = re.findall(initmolreg, datta)
        initmol = initmol[0].split("\n")
        label_xyz = r"(\s*.*(\s*-?\d+\.\d+){3})+"
        molly_init = np.array([])
        insertion = []
        c = 0
        for x, line in enumerate(initmol):
            if re.search(label_xyz, line):
                if re.search(r"\s*[xX]", line):
                    insertion.append(succ)
                else:
                    re.search(label_xyz, line)
                    temp = line.split()[-3:]
                    molly_init = np.append(molly_init, np.array(temp))
                c += 1

        molly_init = molly_init.astype("float64")
        molly_init = np.split(molly_init, len(molly_init) / 3)

        with open("output.dat", "r") as file:
            data = file.readlines()
        for i in range(len(data)):
            molly1 = re.search(molly1_regex, data[i])
            if molly1:
                beg_molly = i + 1
                break
        for i in range(len(data) - beg_molly):
            molly2 = re.search(molly2_regex, data[i + beg_molly])
            if molly2:
                end_molly = i + beg_molly
                break
        label_xyz = r"(\s*.*(\s*-?\d+\.\d+){3})+"
        molly_array = np.array([])
        for line in data[beg_molly:end_molly]:
            if re.search(label_xyz, line):
                temp = line.split()[-3:]
                molly_array = np.append(molly_array, np.array(temp))

        molly_array = molly_array.astype("float64")
        molly_array = np.split(molly_array, len(molly_array) / 3)
        rearrange = []
        for i, initial in enumerate(molly_init):
            for j, final in enumerate(molly_array):
                if sum(np.abs(initial - final)) < 1e-6:

                    rearrange.append(j)
        os.chdir("..")
        return rearrange, insertion

    def reap_energies(self, direc, success_regex, energy_regex):
        if self.options.dir_reap:
            os.chdir("./" + str(direc))
            with open("output.dat", "r") as file:
                data = file.read()
            if not re.search(success_regex, data):
                print("Energy failed at " + os.getcwd())
                raise RuntimeError
            energy = float(re.findall(energy_regex, data)[0])

            os.chdir("..")
        else:
            with open("output." + str(direc) + ".dat", "r") as file:
                data = file.read()
            if not re.search(success_regex, data):
                print("Energy failed at " + os.getcwd())
                raise RuntimeError
            energy = float(re.findall(energy_regex, data)[0])

        return energy

    # def reap_gradients(self, direc, grad_regex1, grad_regex2, shuffle,insertion):
    def reap_gradients(self, direc, grad_regex1, grad_regex2):
        os.chdir("./" + str(direc))
        grad_array = []
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
        # grad_array = grad_array[shuffle]
        # if len(insertion) > 0:
        #    counter = 0
        #    for x in insertion:
        #        grad_array = np.insert(grad_array, x, [0.0, 0.0, 0.0], axis = 0)
        #        counter += 1
        #        grad_array = np.reshape(grad_array, (-1, 3))
        grad_array = grad_array.flatten()
        if not grad1:
            print("Gradient failed at " + os.getcwd())
            raise RuntimeError
        os.chdir("..")
        # raise RuntimeError

        return grad_array
