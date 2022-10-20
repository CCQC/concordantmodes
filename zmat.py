import numpy as np
import os
import shutil
import re
from . import masses
from concordantmodes.int2cart import Int2Cart
from concordantmodes.trans_disp import TransDisp


class Zmat(object):
    def __init__(self, options):
        self.amu_elMass = 5.48579909065 * (10 ** (-4))
        self.disp_tol = 1.0e-14
        self.options = options
        self.Bohr_Ang = 0.529177210903

    def run(self, zmat_name="zmat"):
        # Define some regexes
        zmat_begin_regex = re.compile(r"ZMAT begin")
        zmat_end_regex = re.compile(r"ZMAT end")

        # ZMAT regexes
        #allows for regex of O18 and the like, for heavy isotopes
        first_atom_regex =  re.compile("^\s*([A-Za-z]+[0-9]*)\s*\n")
        second_atom_regex = re.compile("^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s*\n")
        third_atom_regex =  re.compile("^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s+(\d+)\s*\n")
        full_atom_regex =   re.compile("^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*\n")
        # Custom int coord regexes
        bond_regex = re.compile("^\s*(\d+)\s+(\d+)\s*\n")
        angle_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s*\n")
        torsion_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*T\s*\n")
        oop_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*O\s*\n")
        lin_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*L\s*\n")
        linx_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*Lx\s*\n")
        liny_regex = re.compile("^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*Ly\s*\n")

        # Cartesian regexes
        cart_begin_regex = re.compile(r"cart begin")
        cart_end_regex = re.compile(r"cart end")
        s = "[A-Za-z]+[0-9]*\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
        cartesian_regex = re.compile(s)
        s = "([A-Za-z]+[0-9]*)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n"
        cartesian_atom_regex = re.compile(s)
        divider_regex = re.compile("^\s*\-\-\-\s*\n")

        # Read in the ZMAT file
        with open(zmat_name, "r") as file:
            output = file.readlines()

        # Read in the input cartesian coordinates
        self.cartesians_init = []
        self.cartesians_final = []
        cart_range = []
        self.atom_list = []

        for i in range(len(output)):
            beg_cart = re.search(cart_begin_regex, output[i])
            if beg_cart:
                cart_range.append(i)
                break

        for i in range(len(output) - cart_range[0]):
            end_cart = re.search(cart_end_regex, output[i + cart_range[0]])
            if end_cart:
                cart_range.append(i + cart_range[0])
                break

        cart_output = output[cart_range[0] : cart_range[1]].copy()

        for i in range(len(cart_output)):
            divider = re.search(divider_regex, cart_output[i])
            if divider:
                divide_index = i
                break
        if divider:
            cart_output_init = cart_output[:divide_index].copy()
            cart_output_final = cart_output[divide_index + 1 :].copy()
        else:
            cart_output_init = cart_output.copy()

        for i in range(len(cart_output_init)):
            if re.search(cartesian_regex, cart_output_init[i]):
                temp = re.findall(cartesian_regex, cart_output_init[i])
                atom = re.findall(cartesian_atom_regex, cart_output_init[i])
                self.cartesians_init.append(temp[0])
                self.atom_list.append(atom[0])
        self.cartesians_init = np.array(self.cartesians_init).astype(float)

        if divider:
            for i in range(len(cart_output_final)):
                if re.search(cartesian_regex, cart_output_final[i]):
                    temp = re.findall(cartesian_regex, cart_output_final[i])
                    self.cartesians_final.append(temp[0])
            self.cartesians_final = np.array(self.cartesians_final).astype(float)
        else:
            self.cartesians_final = self.cartesians_init.copy()

        if self.options.cart_coords.upper() == "ANGSTROM":
            self.cartesians_init /= self.Bohr_Ang
            self.cartesians_final /= self.Bohr_Ang

        # Slice out the ZMAT from the input
        zmat_range = []

        for i in range(len(output)):
            beg_zmat = re.search(zmat_begin_regex, output[i])
            if beg_zmat:
                zmat_range.append(i)

        for i in range(len(output) - zmat_range[0]):
            end_zmat = re.search(zmat_end_regex, output[i + zmat_range[0]])
            if end_zmat:
                zmat_range.append(i + zmat_range[0])
                break

        zmat_output = output[zmat_range[0] : zmat_range[1]].copy()

        # Initialize necessary lists
        self.bond_indices = []
        self.bond_variables = []
        self.angle_indices = []
        self.angle_variables = []
        self.torsion_indices = []
        self.torsion_variables = []
        self.oop_indices = []
        self.oop_variables = []
        self.lin_indices = []
        self.lin_variables = []
        self.linx_indices = []
        self.linx_variables = []
        self.liny_indices = []
        self.liny_variables = []
        self.variable_dictionary_init = {}
        self.variable_dictionary_final = {}
        self.index_dictionary = {}

        count = 0
        if self.options.coords.upper() == "ZMAT":
            # This code reaps ZMAT data
            for i in range(len(zmat_output)):
                # This case if we are at the first atom of the ZMAT
                if re.search(first_atom_regex, zmat_output[i]) and count < 1:
                    first_index = i
                    count += 1
                # Second atom of the ZMAT, will have one bond term
                if re.search(second_atom_regex, zmat_output[i]):
                    List = re.findall(second_atom_regex, zmat_output[i])[0]
                    self.bond_indices.append([str(i - first_index + 1), List[1]])
                    self.bond_variables.append("R" + str(i - first_index))
                # Third atom of the ZMAT, will have bond and angle term
                if re.search(third_atom_regex, zmat_output[i]):
                    List = re.findall(third_atom_regex, zmat_output[i])[0]
                    self.bond_indices.append([str(i - first_index + 1), List[1]])
                    self.bond_variables.append("R" + str(i - first_index))
                    self.angle_indices.append(
                        [str(i - first_index + 1), List[1], List[2]]
                    )
                    self.angle_variables.append("A" + str(i - first_index))
                # All remaining ZMAT atoms, will have bond, angle, and torsion
                # term
                if re.search(full_atom_regex, zmat_output[i]):
                    List = re.findall(full_atom_regex, zmat_output[i])[0]
                    self.bond_indices.append([str(i - first_index + 1), List[1]])
                    self.bond_variables.append("R" + str(i - first_index))
                    self.angle_indices.append(
                        [str(i - first_index + 1), List[1], List[2]]
                    )
                    self.angle_variables.append("A" + str(i - first_index))
                    self.torsion_indices.append(
                        [str(i - first_index + 1), List[1], List[2], List[3]]
                    )
                    self.torsion_variables.append("D" + str(i - first_index))
        elif self.options.coords.upper() == "REDUNDANT":
            count = 0
            if self.options.interatomic_distance:
                self.bond_indices = np.array([])
                indices = []
                transdisp_inter = TransDisp(
                    1,
                    self,
                    1,
                    1,
                    False,
                    self.disp_tol,
                    np.array([]),
                    self.options,
                    indices,
                )
                inter_atomic_len = np.zeros(
                    (len(self.cartesians_init), len(self.cartesians_init))
                )
                for i in range(len(self.cartesians_init)):
                    for j in range(i):
                        inter_atomic_len[j, i] = transdisp_inter.calc_bond(
                            self.cartesians_init[i], self.cartesians_init[j]
                        )
                        if inter_atomic_len[j, i] < self.options.bond_tol:
                            count += 1
                            self.bond_indices = np.append(
                                self.bond_indices, np.array([str(j + 1), str(i + 1)])
                            )
                            self.bond_variables.append("R" + str(count))
                    self.bond_indices = np.reshape(self.bond_indices, (-1, 2))
                print("Interatomic Distance Matrix:")
                print(inter_atomic_len)
                print("Bond tolerance:")
                print(self.options.bond_tol)
                print("Resulting bond indices:")
                print(self.bond_indices)
            else:
                for i in range(len(zmat_output)):
                    if re.search(bond_regex, zmat_output[i]):
                        count += 1
                        List = re.findall(bond_regex, zmat_output[i])[0]
                        self.bond_indices.append(List)
                        self.bond_variables.append("R" + str(count))
                self.bond_indices = np.array(self.bond_indices)
            # Form all possible angles from bonds
            self.angle_indices = np.array([])
            count = 0
            for i in range(len(self.bond_indices)):
                for j in range(len(self.bond_indices) - i - 1):
                    a = np.setdiff1d(self.bond_indices[i], self.bond_indices[i + j + 1])
                    b = np.intersect1d(
                        self.bond_indices[i], self.bond_indices[i + j + 1]
                    )
                    c = np.setdiff1d(self.bond_indices[i + j + 1], self.bond_indices[i])
                    if len(a) and len(b) and len(c):
                        d = np.array([a[0], b[0], c[0]])
                        self.angle_indices = np.append(self.angle_indices, d)
                        count += 1
                        self.angle_variables.append("A" + str(count))
            self.angle_indices = self.angle_indices.reshape((-1, 3))

            # Form all possible torsions from angles
            self.torsion_indices = np.array([])
            count = 0
            for i in range(len(self.angle_indices)):
                for j in range(len(self.angle_indices) - i - 1):
                    a = np.setdiff1d(
                        self.angle_indices[i], self.angle_indices[i + j + 1]
                    )
                    b = np.intersect1d(
                        self.angle_indices[i], self.angle_indices[i + j + 1]
                    )
                    c = np.setdiff1d(
                        self.angle_indices[i + j + 1], self.angle_indices[i]
                    )
                    if len(a) and len(b) == 2 and len(c):
                        d = np.array([a[0], b[0], b[1], c[0]])
                        self.torsion_indices = np.append(self.torsion_indices, d)
                        count += 1
                        self.torsion_variables.append("D" + str(count))
            self.torsion_indices = self.torsion_indices.reshape((-1, 4))

        elif self.options.coords.upper() == "CUSTOM":
            # This option will allow the user to specify a custom array of
            # internal coordinates.
            Sum = 0
            blank = 0
            for i in range(len(zmat_output)):
                if re.search(bond_regex, zmat_output[i]):
                    List = re.findall(bond_regex, zmat_output[i])[0]
                    self.bond_indices.append(List)
                    self.bond_variables.append(
                        "R" + str(i + 1 - Sum + len(self.bond_variables))
                    )
                elif re.search(angle_regex, zmat_output[i]):
                    List = re.findall(angle_regex, zmat_output[i])[0]
                    self.angle_indices.append(List)
                    self.angle_variables.append(
                        "A" + str(i + 1 - Sum + len(self.angle_variables))
                    )
                elif re.search(torsion_regex, zmat_output[i]):
                    List = re.findall(torsion_regex, zmat_output[i])[0]
                    self.torsion_indices.append(List)
                    self.torsion_variables.append(
                        "D" + str(i + 1 - Sum + len(self.torsion_variables))
                    )
                elif re.search(oop_regex, zmat_output[i]):
                    List = re.findall(oop_regex, zmat_output[i])[0]
                    self.oop_indices.append(List)
                    self.oop_variables.append(
                        "O" + str(i + 1 - Sum + len(self.oop_variables))
                    )
                elif re.search(lin_regex, zmat_output[i]):
                    List = re.findall(lin_regex, zmat_output[i])[0]
                    self.lin_indices.append(List)
                    self.lin_variables.append(
                        "L" + str(i + 1 - Sum + len(self.lin_variables))
                    )
                elif re.search(linx_regex, zmat_output[i]):
                    List = re.findall(linx_regex, zmat_output[i])[0]
                    self.linx_indices.append(List)
                    self.linx_variables.append(
                        "Lx" + str(i + 1 - Sum + len(self.linx_variables))
                    )
                elif re.search(liny_regex, zmat_output[i]):
                    List = re.findall(liny_regex, zmat_output[i])[0]
                    self.liny_indices.append(List)
                    self.liny_variables.append(
                        "Ly" + str(i + 1 - Sum + len(self.liny_variables))
                    )
                else:
                    blank += 1
                Sum = (
                    len(self.bond_variables)
                    + len(self.angle_variables)
                    + len(self.torsion_variables)
                    + len(self.oop_variables)
                    + len(self.lin_variables)
                    + len(self.linx_variables)
                    + len(self.liny_variables)
                    + blank
                )

        # This code utilizes the INTC function from the TransDisp module to
        # calculate the initial variable values from the cartesian
        # coordinates.
        indices = []
        transdisp = TransDisp(
            1, self, 1, 1, False, self.disp_tol, np.array([]), self.options, indices
        )
        I = np.eye(
            len(self.bond_indices)
            + len(self.angle_indices)
            + len(self.torsion_indices)
            + len(self.oop_indices)
            + len(self.lin_indices)
            + len(self.linx_indices)
            + len(self.liny_indices)
        )
        variables1 = transdisp.int_c(self.cartesians_init, I, I)
        variables2 = transdisp.int_c(self.cartesians_final, I, I)
        for i in range(
            len(self.angle_indices)
            + len(self.torsion_indices)
            + len(self.oop_indices)
            + len(self.lin_indices)
            + len(self.linx_indices)
            + len(self.liny_indices)
        ):
            variables1[len(self.bond_indices) + i] *= 180.0 / np.pi
            variables2[len(self.bond_indices) + i] *= 180.0 / np.pi
        variables = np.append(
            self.bond_variables, np.append(self.angle_variables, self.torsion_variables)
        )
        if len(self.oop_variables):
            variables = np.append(variables, self.oop_variables)
        if len(self.lin_variables):
            variables = np.append(variables, self.lin_variables)
        if len(self.linx_variables):
            variables = np.append(variables, self.linx_variables)
        if len(self.liny_variables):
            variables = np.append(variables, self.liny_variables)

        # This code is useful for checking the Redundant coordinate
        # generation process.
        if self.options.coords.upper() == "REDUNDANT":
            indices = []
            for i in range(len(self.bond_indices)):
                indices.append(self.bond_indices[i].tolist())
            for i in range(len(self.angle_indices)):
                indices.append(self.angle_indices[i].tolist())
            for i in range(len(self.torsion_indices)):
                indices.append(self.torsion_indices[i].tolist())
            # print(indices)

        for i in range(len(variables1)):
            self.variable_dictionary_init[variables[i]] = variables1[i]

        if divider:
            for i in range(len(variables2)):
                self.variable_dictionary_final[variables[i]] = variables2[i]
        else:
            self.variable_dictionary_final = self.variable_dictionary_init.copy()
            # And now we must temper the torsion angles! For consistency's sake
            # we will force them to lie between -90 deg and +270 deg.

        # Handle Variable lists separately. First the INIT:
        for i in range(len(self.torsion_variables)):
            condition_1 = (
                float(self.variable_dictionary_init[self.torsion_variables[i]]) <= -90.0
            )
            condition_2 = (
                float(self.variable_dictionary_init[self.torsion_variables[i]]) >= 270.0
            )
            buff = np.floor(
                abs(float(self.variable_dictionary_init[self.torsion_variables[i]]))
                / 360
            )
            if condition_1:
                self.variable_dictionary_init[self.torsion_variables[i]] = float(
                    self.variable_dictionary_init[self.torsion_variables[i]]
                )
                self.variable_dictionary_init[self.torsion_variables[i]] += 360.0 * buff
                if (
                    float(self.variable_dictionary_init[self.torsion_variables[i]])
                    <= -90.0
                ):
                    self.variable_dictionary_init[self.torsion_variables[i]] += 360.0
            if condition_2:
                self.variable_dictionary_init[self.torsion_variables[i]] = float(
                    self.variable_dictionary_init[self.torsion_variables[i]]
                )
                self.variable_dictionary_init[self.torsion_variables[i]] -= 360.0 * buff
                if (
                    float(self.variable_dictionary_init[self.torsion_variables[i]])
                    >= 270.0
                ):
                    self.variable_dictionary_init[self.torsion_variables[i]] -= 360.0
        # Then the Final. This can probably be structured more elegantly, but this works and isn't too computationally demanding.
        for i in range(len(self.torsion_variables)):
            condition_1 = (
                float(self.variable_dictionary_final[self.torsion_variables[i]])
                <= -90.0
            )
            condition_2 = (
                float(self.variable_dictionary_final[self.torsion_variables[i]])
                >= 270.0
            )
            buff = np.floor(
                abs(float(self.variable_dictionary_final[self.torsion_variables[i]]))
                / 360
            )
            if condition_1:
                self.variable_dictionary_final[self.torsion_variables[i]] = float(
                    self.variable_dictionary_final[self.torsion_variables[i]]
                )
                self.variable_dictionary_final[self.torsion_variables[i]] += (
                    360.0 * buff
                )
                if (
                    float(self.variable_dictionary_final[self.torsion_variables[i]])
                    <= -90.0
                ):
                    self.variable_dictionary_final[self.torsion_variables[i]] += 360.0
            if condition_2:
                self.variable_dictionary_final[self.torsion_variables[i]] = float(
                    self.variable_dictionary_final[self.torsion_variables[i]]
                )
                self.variable_dictionary_final[self.torsion_variables[i]] -= (
                    360.0 * buff
                )
                if (
                    float(self.variable_dictionary_final[self.torsion_variables[i]])
                    >= 270.0
                ):
                    self.variable_dictionary_final[self.torsion_variables[i]] -= 360.0

        # The masses are assigned to the respective atom from the masses.py file
        self.masses = [masses.get_mass(label) for label in self.atom_list]
        for i in range(len(self.masses)):
            self.masses[i] = self.masses[i] / self.amu_elMass

        self.mass_weight = np.diag(np.array(self.masses).repeat(3))

        zmat_shift_a = 0
        zmat_shift_d = 0

        if self.options.coords.upper() == "ZMAT":
            zmat_shift_a = 1
            zmat_shift_d = 2

        # Append all indices to index_dictionary
        for i in range(len(self.bond_indices)):
            self.index_dictionary["R" + str(i + 1)] = self.bond_indices[i]
        for i in range(len(self.angle_indices)):
            self.index_dictionary["A" + str(i + zmat_shift_a + 1)] = self.angle_indices[
                i
            ]
        for i in range(len(self.torsion_indices)):
            self.index_dictionary[
                "D" + str(i + zmat_shift_d + 1)
            ] = self.torsion_indices[i]
        for i in range(len(self.oop_indices)):
            self.index_dictionary["O" + str(i + 1)] = self.oop_indices[i]
        for i in range(len(self.lin_indices)):
            self.index_dictionary["L" + str(i + 1)] = self.lin_indices[i]
        for i in range(len(self.linx_indices)):
            self.index_dictionary["Lx" + str(i + 1)] = self.linx_indices[i]
        for i in range(len(self.liny_indices)):
            self.index_dictionary["Ly" + str(i + 1)] = self.liny_indices[i]

        # Print off the internal coordinate and its value in Bohr/Degree
        print("Initial Geometric Internal Coordinate Values:")
        for i in range(len(variables)):
            print(
                str(self.index_dictionary[variables[i]])
                + " "
                + variables[i]
                + " = "
                + str(self.variable_dictionary_init[variables[i]])
            )
            # if self.options.coords.upper() == "REDUNDANT":
            # print(indices[i])
        print("Final Geometric Internal Coordinate Values:")
        for i in range(len(variables)):
            print(
                str(self.index_dictionary[variables[i]])
                + " "
                + variables[i]
                + " = "
                + str(self.variable_dictionary_final[variables[i]])
            )
            # if self.options.coords.upper() == "REDUNDANT":
            # print(indices[i])
        print("Final - Initial Geometric Internal Coordinate Values:")
        for i in range(len(variables)):
            print(
                variables[i]
                + " = "
                + str(
                    self.variable_dictionary_final[variables[i]]
                    - self.variable_dictionary_init[variables[i]]
                )
            )
            # if self.options.coords.upper() == "REDUNDANT":
            # print(indices[i])
        if self.options.geom_check:
            Sum = 0
            for i in range(len(self.bond_indices)):
                Sum += (
                    self.variable_dictionary_final[variables[i]]
                    - self.variable_dictionary_init[variables[i]]
                ) ** 2

            # print("squared sum: ")
            # print(Sum)
            # print("# of bonds:")
            # print(len(self.bond_indices))
            # print("RMSD:")
            # print(np.sqrt(Sum / len(self.bond_indices)))
        # Calculate Cartesians using ZMATs: Sadly this will have to go on the backburner.
        # The cartesians must match those used to generate the cartesian force constants or you're gonna have a bad time.

        # This could be used for initHess option?

        # cartInit = int2cart(self,self.variable_dictionary_init)
        # cartFinal = int2cart(self,self.variable_dictionary_final)
        # cartInit.run()
        # cartFinal.run()
        # self.cartesians_init = cartInit.Carts
        # self.cartesians_final = cartFinal.Carts
        # print(self.cartesians_init)
        # print(self.cartesians_final)

        # print(self.cartesians_init)
        # print(self.cartesians_final)
        # print(self.bond_variables)
        # print(self.torsion_indices)
