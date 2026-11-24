import numpy as np
import os
import shutil
import re
import collections
from . import masses
from qcelemental.covalent_radii import CovalentRadii
from concordantmodes.int2cart import Int2Cart
from concordantmodes.transf_disp import TransfDisp
import logging
from concordantmodes.errors import CMAError
from concordantmodes.log import debug_print, info_print

logger = logging.getLogger(__name__)


class Zmat:
    """
    Parse, construct, and analyze molecular internal coordinate systems.

    The ``Zmat`` class reads molecular geometry information from a ZMAT-style
    input file, constructs internal coordinate definitions, evaluates their
    numerical values from Cartesian coordinates, and provides mappings between
    coordinate labels, atom indices, and coordinate values.

    The class supports three coordinate generation modes:

    * ``ZMAT`` – Traditional Z-matrix coordinates generated directly from
      the atom connectivity specified in the Z-matrix section.
    * ``DELOCALIZED`` – Redundant internal coordinates generated
      automatically from molecular connectivity determined either from
      user-supplied bonds or covalent radii analysis.
    * ``CUSTOM`` – User-defined internal coordinates, including support
      for centroid-based coordinates and center-of-mass distances.

    Supported internal coordinate types include:

    * Bond stretches
    * Bond angles
    * Dihedral/torsional angles
    * Out-of-plane bends
    * Linear bending coordinates
    * Linear bend x/y components
    * Center-of-mass separation coordinates
    * Centroid-based coordinates

    The class reads one or two Cartesian geometries from the input file.
    When two geometries are supplied, the first is treated as the initial
    geometry and the second as the final geometry. Internal coordinate
    values are evaluated for both geometries and stored for comparison.

    Attributes
    ----------
    options : object
        User options controlling coordinate generation, units,
        topology analysis, and parsing behavior.

    atom_list : list[str]
        Atomic labels read from the Cartesian geometry section.

    cartesians_b : ndarray, shape (N, 3)
        Initial Cartesian coordinates in atomic units.

    cartesians_a : ndarray, shape (N, 3)
        Final Cartesian coordinates in atomic units.

    masses : list[float]
        Atomic masses converted to electron-mass units.

    mass_weight : ndarray, shape (3N, 3N)
        Diagonal mass-weighting matrix.

    bond_indices : ndarray
        Bond coordinate atom index definitions.

    angle_indices : ndarray
        Angle coordinate atom index definitions.

    torsion_indices : ndarray
        Torsional coordinate atom index definitions.

    oop_indices : ndarray
        Out-of-plane bend coordinate definitions.

    lin_indices : ndarray
        Linear bend coordinate definitions.

    linx_indices : ndarray
        Linear bend x-component definitions.

    liny_indices : ndarray
        Linear bend y-component definitions.

    rcom_indices : ndarray
        Center-of-mass distance coordinate definitions.

    variables : ndarray
        Ordered list of internal coordinate labels.

    variable_dictionary_b : dict
        Internal coordinate values for the initial geometry.

    variable_dictionary_a : dict
        Internal coordinate values for the final geometry.

    index_dictionary : dict
        Mapping between coordinate labels and their corresponding
        atom index definitions.

    Methods
    -------
    run(zmat_name='zmat')
        Execute the complete ZMAT processing workflow.

    zmat_read(zmat_name)
        Read Cartesian coordinates and Z-matrix information from
        an input file.

    zmat_process(zmat_output)
        Construct internal coordinate definitions from parsed input.

    zmat_calc()
        Compute internal coordinate values from Cartesian geometries.

    zmat_compile()
        Build lookup dictionaries for coordinate labels and indices.

    zmat_print()
        Print initial, final, and displacement values for all
        internal coordinates.

    red_mass(indices)
        Compute the reciprocal reduced mass contribution for a
        coordinate definition.

    np_contains(array1, array2, tor=False)
        Determine whether a coordinate definition already exists
        within a coordinate list.

    Notes
    -----
    Cartesian coordinates may be supplied in either Angstroms or
    Bohr depending on the value of ``options.cart_coords``. All
    internal calculations are performed in atomic units.

    For ``DELOCALIZED`` coordinates, molecular connectivity can be
    determined automatically using covalent radii from
    ``qcelemental.covalent_radii.CovalentRadii``. From this
    connectivity graph, redundant bonds, angles, torsions, and
    out-of-plane bends are generated automatically.

    When topology analysis is enabled, the class can additionally
    generate and analyze molecular graph walks and ring cycles of
    increasing length.
    """

    def __init__(self, options):
        self.amu_elMass = 5.48579909065 * (10 ** (-4))
        self.disp_tol = 1.0e-14
        self.options = options
        self.Bohr_Ang = 0.529177210903

    def run(self, zmat_name="zmat"):
        # Read in the ZMAT file
        zmat_output = self.zmat_read(zmat_name)

        # Process the read in information
        self.zmat_process(zmat_output)

        # Calculate internal coordinate values from reference cartesian coordinates
        self.zmat_calc()

        #
        self.zmat_compile()

        #
        self.zmat_print()

    def zmat_read(self, zmat_name):
        # Define some regexes
        self.zmat_begin_regex = re.compile(r"ZMAT begin")
        self.zmat_end_regex = re.compile(r"ZMAT end")

        # ZMAT regexes
        self.first_atom_regex = re.compile(r"^\s*([A-Za-z]+[0-9]*)\s*\n")
        self.second_atom_regex = re.compile(r"^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s*\n")
        self.third_atom_regex = re.compile(
            r"^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s+(\d+)\s*\n"
        )
        self.full_atom_regex = re.compile(
            r"^\s*([A-Za-z]+[0-9]*)\s+(\d+)\s+(\d+)\s+(\d+)\s*\n"
        )
        # Custom int coord regexes
        self.bond_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s*\n")
        self.angle_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*\n")
        self.torsion_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*T\s*\n")
        self.oop_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*O\s*\n")
        self.lin_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*L\s*\n")
        self.linx_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*Lx\s*\n")
        self.liny_regex = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*Ly\s*\n")
        self.rcom_regex1 = re.compile(r";\s*((\d+\s+)+)\s*;\s*((\d+\s+)+)\s*Rc")
        self.rcom_regex2 = re.compile(r"\s*(\d+)")

        # Centroid regexes
        self.centroid_regex1 = re.compile(r";")
        self.centroid_regex2 = re.compile(r"\s*(\d+)")
        self.bond_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)\n"
        )
        self.angle_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)\n"
        )
        self.torsion_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)T\s*\n"
        )
        self.oop_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)O\s*\n"
        )
        self.lin_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)L\s*\n"
        )
        self.linx_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)Lx\s*\n"
        )
        self.liny_centroid_regex = re.compile(
            r"^\s*(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)(;\s*(\d+\s+)*\d+\s*)Ly\s*\n"
        )

        # Cartesian regexes
        self.cart_begin_regex = re.compile(r"cart begin")
        self.cart_end_regex = re.compile(r"cart end")
        s = r"[A-Za-z]+[0-9]*\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
        self.cartesian_regex = re.compile(s)
        s = r"([A-Za-z]+[0-9]*)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n"
        self.cartesian_atom_regex = re.compile(s)
        self.divider_regex = re.compile(r"^\s*\-\-\-\s*\n")
        if not os.path.exists(zmat_name):
            raise CMAError(
                "The geometry file '{}' was not found in {}.".format(
                    zmat_name, os.getcwd()
                )
            )
        with open(zmat_name, "r") as file:
            output = file.readlines()
        # Every regex below expects a trailing newline, including on the last line.
        output = [line if line.endswith("\n") else line + "\n" for line in output]
        self.check_input(output, zmat_name)
        # Read in the input cartesian coordinates
        self.cartesians_b = []
        self.cartesians_a = []
        cart_range = []
        self.atom_list = []

        for i in range(len(output)):
            beg_cart = re.search(self.cart_begin_regex, output[i])
            if beg_cart:
                cart_range.append(i)
                break

        for i in range(len(output) - cart_range[0]):
            end_cart = re.search(self.cart_end_regex, output[i + cart_range[0]])
            if end_cart:
                cart_range.append(i + cart_range[0])
                break

        self.cart_range = cart_range
        cart_output = output[cart_range[0] : cart_range[1]].copy()

        for i in range(len(cart_output)):
            self.divider = re.search(self.divider_regex, cart_output[i])
            if self.divider:
                divide_index = i
                break
        if self.divider:
            cart_output_b = cart_output[:divide_index].copy()
            cart_output_a = cart_output[divide_index + 1 :].copy()
        else:
            cart_output_b = cart_output.copy()

        for i in range(len(cart_output_b)):
            if re.search(self.cartesian_regex, cart_output_b[i]):
                temp = re.findall(self.cartesian_regex, cart_output_b[i])
                atom = re.findall(self.cartesian_atom_regex, cart_output_b[i])
                self.cartesians_b.append(temp[0])
                self.atom_list.append(atom[0].upper())
        self.cartesians_b = np.array(self.cartesians_b).astype(float)

        # The masses are assigned to the respective atom from the masses.py file
        self.masses = [masses.get_mass(label) for label in self.atom_list]
        for i in range(len(self.masses)):
            self.masses[i] = self.masses[i] / self.amu_elMass

        self.mass_weight = np.diag(np.array(self.masses).repeat(3))

        if self.divider:
            for i in range(len(cart_output_a)):
                if re.search(self.cartesian_regex, cart_output_a[i]):
                    temp = re.findall(self.cartesian_regex, cart_output_a[i])
                    self.cartesians_a.append(temp[0])
            self.cartesians_a = np.array(self.cartesians_a).astype(float)
        else:
            self.cartesians_a = self.cartesians_b.copy()

        if self.options.cart_coords.upper() == "ANGSTROM":
            self.cartesians_b /= self.Bohr_Ang
            self.cartesians_a /= self.Bohr_Ang

        zmat_output = ""
        # Slice out the ZMAT from the input
        if not self.options.covalent_radii:
            zmat_range = []

            for i in range(len(output)):
                beg_zmat = re.search(self.zmat_begin_regex, output[i])
                if beg_zmat:
                    zmat_range.append(i)

            for i in range(len(output) - zmat_range[0]):
                end_zmat = re.search(self.zmat_end_regex, output[i + zmat_range[0]])
                if end_zmat:
                    zmat_range.append(i + zmat_range[0])
                    break

            zmat_output = output[zmat_range[0] + 1 : zmat_range[1]].copy()

        return zmat_output

    def check_input(self, output, zmat_name="zmat"):
        """Check the zmat file's layout before parsing and report the first
        problem with its line number (instead of failing deep in the parser)."""

        def where(i):
            return "{} line {} ({!r})".format(zmat_name, i + 1, output[i].rstrip("\n"))

        def find(regex, start=0):
            for i in range(start, len(output)):
                if regex.search(output[i]):
                    return i
            return None

        if not any(line.strip() for line in output):
            raise CMAError(
                "{} is empty. It needs a 'cart begin' ... 'cart end' block with "
                "the geometry and a 'ZMAT begin' ... 'ZMAT end' block with the "
                "coordinates; see the examples directory.".format(zmat_name)
            )

        cb = find(self.cart_begin_regex)
        if cb is None:
            raise CMAError(
                "{} has no 'cart begin' line. Put the Cartesian geometry between "
                "'cart begin' and 'cart end' lines.".format(zmat_name)
            )
        ce = find(self.cart_end_regex, cb + 1)
        if ce is None:
            raise CMAError(
                "{}: 'cart begin' on line {} is never closed by a 'cart end' "
                "line.".format(zmat_name, cb + 1)
            )

        n_atoms = [0]
        for i in range(cb + 1, ce):
            line = output[i]
            if not line.strip():
                continue
            if self.divider_regex.search(line):
                n_atoms.append(0)
                continue
            if not self.cartesian_regex.search(line):
                raise CMAError(
                    "{} is not a valid Cartesian line. Expected an element symbol and "
                    "three coordinates written with a decimal point, e.g. "
                    "'O  0.000000  0.000000  -0.125505'.".format(where(i))
                )
            n_atoms[-1] += 1
        if n_atoms[0] == 0:
            raise CMAError(
                "{}: the cart block (lines {}-{}) contains no atoms.".format(
                    zmat_name, cb + 1, ce + 1
                )
            )
        if len(n_atoms) > 2:
            raise CMAError(
                "{}: the cart block has more than one '---' divider; use at "
                "most one (level-B geometry, ---, level-A geometry).".format(zmat_name)
            )
        if len(n_atoms) == 2 and n_atoms[1] != n_atoms[0]:
            raise CMAError(
                "{}: the two geometries in the cart block have different numbers of atoms "
                "({} before '---', {} after).".format(zmat_name, n_atoms[0], n_atoms[1])
            )
        natom = n_atoms[0]

        if self.options.covalent_radii:
            return

        zb = find(self.zmat_begin_regex)
        if zb is None:
            raise CMAError(
                "{} has no 'ZMAT begin' line. List the internal coordinates "
                "between 'ZMAT begin' and 'ZMAT end' (or set covalent_radii = "
                "True to find bonds automatically).".format(zmat_name)
            )
        ze = find(self.zmat_end_regex, zb + 1)
        if ze is None:
            raise CMAError(
                "{}: 'ZMAT begin' on line {} is never closed by a 'ZMAT end' "
                "line.".format(zmat_name, zb + 1)
            )

        # Remembered for check_coordinate_lines (run when the block is parsed).
        self._check_ctx = (zmat_name, zb, natom)

    def check_coordinate_lines(self, zmat_output):
        """Check every line of the ZMAT block against the forms valid for the
        chosen coords mode, and every atom number against the cart block."""
        ctx = getattr(self, "_check_ctx", None)
        if ctx is None:
            return
        zmat_name, zb, natom = ctx

        def where(i, line):
            return "{} line {} ({!r})".format(zmat_name, i + 1, line.rstrip("\n"))

        if self.options.coords.upper() == "ZMAT":
            forms = [
                self.first_atom_regex,
                self.second_atom_regex,
                self.third_atom_regex,
                self.full_atom_regex,
            ]
            hint = (
                "In coords = 'ZMAT' mode each line is an atom and the atoms it is "
                "connected to, e.g. 'O', 'H 1', 'H 1 2'."
            )
        else:
            forms = [
                self.bond_regex,
                self.angle_regex,
                self.torsion_regex,
                self.oop_regex,
                self.lin_regex,
                self.linx_regex,
                self.liny_regex,
                self.rcom_regex1,
                self.bond_centroid_regex,
                self.angle_centroid_regex,
                self.torsion_centroid_regex,
                self.oop_centroid_regex,
                self.lin_centroid_regex,
                self.linx_centroid_regex,
                self.liny_centroid_regex,
            ]
            hint = (
                "Valid lines are 'i j' (bond), 'i j k' (angle), 'i j k l T' (torsion), "
                "'i j k l O' (out-of-plane), 'i j k l L|Lx|Ly' (linear bends), with "
                "atom numbers from the cart block; groups of atoms start with ';', e.g. "
                "'; 1 ; 3 4 5' for a bond to the centroid of atoms 3-5."
            )
        for k, line in enumerate(zmat_output):
            i = zb + 1 + k
            if not line.strip():
                continue
            if not any(f.search(line) for f in forms):
                raise CMAError(
                    "{} is not a recognised coordinate line. {}".format(
                        where(i, line), hint
                    )
                )
            numbers = [int(x) for x in re.findall(r"\b\d+\b", line)]
            bad = [n for n in numbers if n < 1 or n > natom]
            if bad:
                raise CMAError(
                    "{} refers to atom {} but the cart block has {} atoms (atoms are "
                    "numbered from 1).".format(where(i, line), bad[0], natom)
                )

    def zmat_process(self, zmat_output):
        self.check_coordinate_lines(zmat_output)
        # Initialize necessary lists
        self.bond_indices = np.array([], dtype=object)
        self.bond_variables = []
        self.angle_indices = np.array([], dtype=object)
        self.angle_variables = []
        self.torsion_indices = np.array([], dtype=object)
        self.torsion_variables = []
        self.oop_indices = np.array([], dtype=object)
        self.oop_variables = []
        self.lin_indices = np.array([], dtype=object)
        self.lin_variables = []
        self.linx_indices = np.array([], dtype=object)
        self.linx_variables = []
        self.liny_indices = np.array([], dtype=object)
        self.liny_variables = []
        self.rcom_indices = np.array([], dtype=object)
        self.rcom_variables = []
        self.variable_dictionary_b = {}
        self.variable_dictionary_a = {}
        self.index_dictionary = {}
        self.reduced_masses = np.array([])

        count = 0
        if self.options.coords.upper() == "ZMAT":
            # This code reaps ZMAT data
            for i in range(len(zmat_output)):
                # This case if we are at the first atom of the ZMAT
                if re.search(self.first_atom_regex, zmat_output[i]) and count < 1:
                    first_index = i
                    count += 1
                # Second atom of the ZMAT, will have one bond term
                if re.search(self.second_atom_regex, zmat_output[i]):
                    List = re.findall(self.second_atom_regex, zmat_output[i])[0]
                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1]], dtype=object
                    )
                    self.bond_variables.append("R" + str(i - first_index))
                # Third atom of the ZMAT, will have bond and angle term
                if re.search(self.third_atom_regex, zmat_output[i]):
                    List = re.findall(self.third_atom_regex, zmat_output[i])[0]
                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1]], dtype=object
                    )
                    self.bond_variables.append("R" + str(i - first_index))
                    self.angle_indices = np.append(self.angle_indices, 0)
                    self.angle_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1], List[2]], dtype=object
                    )
                    self.angle_variables.append("A" + str(i - first_index))
                # All remaining ZMAT atoms, will have bond, angle, and torsion
                # term
                if re.search(self.full_atom_regex, zmat_output[i]):
                    List = re.findall(self.full_atom_regex, zmat_output[i])[0]
                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1]], dtype=object
                    )
                    self.bond_variables.append("R" + str(i - first_index))

                    self.angle_indices = np.append(self.angle_indices, 0)
                    self.angle_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1], List[2]], dtype=object
                    )
                    self.angle_variables.append("A" + str(i - first_index))

                    self.torsion_indices = np.append(self.torsion_indices, 0)
                    self.torsion_indices[-1] = np.array(
                        [str(i - first_index + 1), List[1], List[2], List[3]],
                        dtype=object,
                    )
                    self.torsion_variables.append("D" + str(i - first_index))
        elif self.options.coords.upper() == "DELOCALIZED":

            # Form all possible bonds, whether from qcelemental or manually input
            self._build_bonds(zmat_output)

            # Form all possible angles from bonds
            self._build_angles()

            # Form all possible torsions from angles
            self._build_four_center()

            # Perform a topological analysis, this is the first step to the
            # automatic generation of Natural Internal Coordinates
            if self.options.topo_analysis:
                self._generate_topology()

        elif self.options.coords.upper() == "CUSTOM":
            # This option will allow the user to specify a custom array of
            # internal coordinates.
            Sum = 0
            blank = 0
            for i in range(len(zmat_output)):
                if re.search(self.bond_regex, zmat_output[i]):
                    List = re.findall(self.bond_regex, zmat_output[i])[0]
                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(List, dtype=object)
                    self.bond_variables.append(
                        "R" + str(i + 1 - Sum + len(self.bond_variables))
                    )
                elif re.search(self.bond_centroid_regex, zmat_output[i]):
                    List = re.findall(self.bond_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff

                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(List, dtype=object)
                    self.bond_variables.append(
                        "R" + str(i + 1 - Sum + len(self.bond_variables))
                    )
                elif re.search(self.angle_regex, zmat_output[i]):
                    List = re.findall(self.angle_regex, zmat_output[i])[0]
                    self.angle_indices = np.append(self.angle_indices, 0)
                    self.angle_indices[-1] = np.array(List, dtype=object)
                    self.angle_variables.append(
                        "A" + str(i + 1 - Sum + len(self.angle_variables))
                    )
                elif re.search(self.angle_centroid_regex, zmat_output[i]):
                    List = re.findall(self.angle_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.angle_indices = np.append(self.angle_indices, 0)
                    self.angle_indices[-1] = np.array(List, dtype=object)
                    self.angle_variables.append(
                        "A" + str(i + 1 - Sum + len(self.angle_variables))
                    )
                elif re.search(self.torsion_regex, zmat_output[i]):
                    List = re.findall(self.torsion_regex, zmat_output[i])[0]
                    self.torsion_indices = np.append(self.torsion_indices, 0)
                    self.torsion_indices[-1] = np.array(List, dtype=object)
                    self.torsion_variables.append(
                        "D" + str(i + 1 - Sum + len(self.torsion_variables))
                    )
                elif re.search(self.torsion_centroid_regex, zmat_output[i]):
                    List = re.findall(self.torsion_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.torsion_indices = np.append(self.torsion_indices, 0)
                    self.torsion_indices[-1] = np.array(List, dtype=object)
                    self.torsion_variables.append(
                        "D" + str(i + 1 - Sum + len(self.torsion_variables))
                    )
                elif re.search(self.oop_regex, zmat_output[i]):
                    List = re.findall(self.oop_regex, zmat_output[i])[0]
                    self.oop_indices = np.append(self.oop_indices, 0)
                    self.oop_indices[-1] = np.array(List, dtype=object)
                    self.oop_variables.append(
                        "O" + str(i + 1 - Sum + len(self.oop_variables))
                    )
                elif re.search(self.oop_centroid_regex, zmat_output[i]):
                    List = re.findall(self.oop_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.oop_indices = np.append(self.oop_indices, 0)
                    self.oop_indices[-1] = np.array(List, dtype=object)
                    self.oop_variables.append(
                        "O" + str(i + 1 - Sum + len(self.oop_variables))
                    )
                elif re.search(self.lin_regex, zmat_output[i]):
                    List = re.findall(self.lin_regex, zmat_output[i])[0]
                    self.lin_indices = np.append(self.lin_indices, 0)
                    self.lin_indices[-1] = np.array(List, dtype=object)
                    self.lin_variables.append(
                        "L" + str(i + 1 - Sum + len(self.lin_variables))
                    )
                elif re.search(self.lin_centroid_regex, zmat_output[i]):
                    List = re.findall(self.lin_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.lin_indices = np.append(self.lin_indices, 0)
                    self.lin_indices[-1] = np.array(List, dtype=object)
                    self.lin_variables.append(
                        "L" + str(i + 1 - Sum + len(self.lin_variables))
                    )
                elif re.search(self.linx_regex, zmat_output[i]):
                    List = re.findall(self.linx_regex, zmat_output[i])[0]
                    self.linx_indices = np.append(self.linx_indices, 0)
                    self.linx_indices[-1] = np.array(List, dtype=object)
                    self.linx_variables.append(
                        "Lx" + str(i + 1 - Sum + len(self.linx_variables))
                    )
                elif re.search(self.linx_centroid_regex, zmat_output[i]):
                    List = re.findall(self.linx_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.linx_indices = np.append(self.linx_indices, 0)
                    self.linx_indices[-1] = np.array(List, dtype=object)
                    self.linx_variables.append(
                        "Lx" + str(i + 1 - Sum + len(self.linx_variables))
                    )
                elif re.search(self.liny_regex, zmat_output[i]):
                    List = re.findall(self.liny_regex, zmat_output[i])[0]
                    self.liny_indices = np.append(self.liny_indices, 0)
                    self.liny_indices[-1] = np.array(List, dtype=object)
                    self.liny_variables.append(
                        "Ly" + str(i + 1 - Sum + len(self.liny_variables))
                    )
                elif re.search(self.liny_centroid_regex, zmat_output[i]):
                    List = re.findall(self.liny_centroid_regex, zmat_output[i])[0]
                    ListBuff = []
                    for j in List:
                        if re.search(self.centroid_regex1, j):
                            SubList = re.findall(self.centroid_regex2, j)
                            ListBuff.append(SubList)
                    List = ListBuff
                    self.liny_indices = np.append(self.liny_indices, 0)
                    self.liny_indices[-1] = np.array(List, dtype=object)
                    self.liny_variables.append(
                        "Ly" + str(i + 1 - Sum + len(self.liny_variables))
                    )
                elif re.search(self.rcom_regex1, zmat_output[i]):
                    List = re.findall(self.rcom_regex1, zmat_output[i])[0]
                    List1 = re.findall(self.rcom_regex2, List[0])
                    List2 = re.findall(self.rcom_regex2, List[2])
                    List = [List1, List2]
                    debug_print(logger, "RCOM List:")
                    debug_print(logger, List)
                    self.rcom_indices = np.append(self.rcom_indices, 0)
                    self.rcom_indices[-1] = np.array(List, dtype=object)
                    self.rcom_variables.append(
                        "Rc" + str(i + 1 - Sum + len(self.rcom_variables))
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
                    + len(self.rcom_variables)
                    + blank
                )

    def zmat_calc(self):
        # This code utilizes the INTC function from the TransfDisp module to
        # calculate the initial variable values from the cartesian
        # coordinates.
        indices = []
        transdisp = TransfDisp(
            None, self, 1, False, np.array([]), self.options, indices, None
        )
        I = np.eye(
            len(self.bond_indices)
            + len(self.rcom_indices)
            + len(self.angle_indices)
            + len(self.torsion_indices)
            + len(self.oop_indices)
            + len(self.lin_indices)
            + len(self.linx_indices)
            + len(self.liny_indices)
        )
        self.variables1 = transdisp.int_c(self.cartesians_b, I, I)
        self.variables2 = transdisp.int_c(self.cartesians_a, I, I)

        for i in range(
            +len(self.angle_indices)
            + len(self.torsion_indices)
            + len(self.oop_indices)
            + len(self.lin_indices)
            + len(self.linx_indices)
            + len(self.liny_indices)
        ):
            self.variables1[len(self.bond_indices) + len(self.rcom_indices) + i] *= (
                180.0 / np.pi
            )
            self.variables2[len(self.bond_indices) + len(self.rcom_indices) + i] *= (
                180.0 / np.pi
            )
        self.variables = np.array(self.bond_variables)
        if len(self.rcom_variables):
            self.variables = np.append(self.variables, self.rcom_variables)
        if len(self.angle_variables):
            self.variables = np.append(self.variables, self.angle_variables)
        if len(self.torsion_variables):
            self.variables = np.append(self.variables, self.torsion_variables)
        if len(self.oop_variables):
            self.variables = np.append(self.variables, self.oop_variables)
        if len(self.lin_variables):
            self.variables = np.append(self.variables, self.lin_variables)
        if len(self.linx_variables):
            self.variables = np.append(self.variables, self.linx_variables)
        if len(self.liny_variables):
            self.variables = np.append(self.variables, self.liny_variables)

        # This code is useful for checking the Redundant coordinate
        # generation process.
        if self.options.coords.upper() == "DELOCALIZED":
            indices = []
            for i in range(len(self.bond_indices)):
                indices.append(self.bond_indices[i].tolist())
            for i in range(len(self.angle_indices)):
                indices.append(self.angle_indices[i].tolist())
            for i in range(len(self.torsion_indices)):
                indices.append(self.torsion_indices[i].tolist())

        for i in range(len(self.variables1)):
            self.variable_dictionary_b[self.variables[i]] = self.variables1[i]

        if self.divider:
            for i in range(len(self.variables2)):
                self.variable_dictionary_a[self.variables[i]] = self.variables2[i]
        else:
            self.variable_dictionary_a = self.variable_dictionary_b.copy()
            # And now we must temper the torsion angles! For consistency's sake
            # we will force them to lie between -90 deg and +270 deg.

        # Handle Variable lists separately. First the INIT:
        for i in range(len(self.torsion_variables)):
            condition_1 = (
                float(self.variable_dictionary_b[self.torsion_variables[i]]) <= -90.0
            )
            condition_2 = (
                float(self.variable_dictionary_b[self.torsion_variables[i]]) >= 270.0
            )
            buff = np.floor(
                abs(float(self.variable_dictionary_b[self.torsion_variables[i]])) / 360
            )
            if condition_1:
                self.variable_dictionary_b[self.torsion_variables[i]] = float(
                    self.variable_dictionary_b[self.torsion_variables[i]]
                )
                self.variable_dictionary_b[self.torsion_variables[i]] += 360.0 * buff
                if (
                    float(self.variable_dictionary_b[self.torsion_variables[i]])
                    <= -90.0
                ):
                    self.variable_dictionary_b[self.torsion_variables[i]] += 360.0
            if condition_2:
                self.variable_dictionary_b[self.torsion_variables[i]] = float(
                    self.variable_dictionary_b[self.torsion_variables[i]]
                )
                self.variable_dictionary_b[self.torsion_variables[i]] -= 360.0 * buff
                if (
                    float(self.variable_dictionary_b[self.torsion_variables[i]])
                    >= 270.0
                ):
                    self.variable_dictionary_b[self.torsion_variables[i]] -= 360.0

        # Then the Final. This can probably be structured more elegantly, but this works and isn't too computationally demanding.
        for i in range(len(self.torsion_variables)):
            condition_1 = (
                float(self.variable_dictionary_a[self.torsion_variables[i]]) <= -90.0
            )
            condition_2 = (
                float(self.variable_dictionary_a[self.torsion_variables[i]]) >= 270.0
            )
            buff = np.floor(
                abs(float(self.variable_dictionary_a[self.torsion_variables[i]])) / 360
            )
            if condition_1:
                self.variable_dictionary_a[self.torsion_variables[i]] = float(
                    self.variable_dictionary_a[self.torsion_variables[i]]
                )
                self.variable_dictionary_a[self.torsion_variables[i]] += 360.0 * buff
                if (
                    float(self.variable_dictionary_a[self.torsion_variables[i]])
                    <= -90.0
                ):
                    self.variable_dictionary_a[self.torsion_variables[i]] += 360.0
            if condition_2:
                self.variable_dictionary_a[self.torsion_variables[i]] = float(
                    self.variable_dictionary_a[self.torsion_variables[i]]
                )
                self.variable_dictionary_a[self.torsion_variables[i]] -= 360.0 * buff
                if (
                    float(self.variable_dictionary_a[self.torsion_variables[i]])
                    >= 270.0
                ):
                    self.variable_dictionary_a[self.torsion_variables[i]] -= 360.0

    def zmat_compile(self):
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
            self.index_dictionary["D" + str(i + zmat_shift_d + 1)] = (
                self.torsion_indices[i]
            )
        for i in range(len(self.oop_indices)):
            self.index_dictionary["O" + str(i + 1)] = self.oop_indices[i]
        for i in range(len(self.lin_indices)):
            self.index_dictionary["L" + str(i + 1)] = self.lin_indices[i]
        for i in range(len(self.linx_indices)):
            self.index_dictionary["Lx" + str(i + 1)] = self.linx_indices[i]
        for i in range(len(self.liny_indices)):
            self.index_dictionary["Ly" + str(i + 1)] = self.liny_indices[i]
        for i in range(len(self.rcom_indices)):
            self.index_dictionary["Rc" + str(i + 1)] = self.rcom_indices[i]

    def zmat_print(self):
        # Print off the internal coordinate and its value in Bohr/Degree
        debug_print(logger, "Initial Geometric Internal Coordinate Values:")
        for i in range(len(self.variables)):
            debug_print(
                logger,
                str(self.index_dictionary[self.variables[i]])
                + " "
                + self.variables[i]
                + " = "
                + str(self.variable_dictionary_b[self.variables[i]]),
            )
        debug_print(logger, "Final Geometric Internal Coordinate Values:")
        for i in range(len(self.variables)):
            debug_print(
                logger,
                str(self.index_dictionary[self.variables[i]])
                + " "
                + self.variables[i]
                + " = "
                + str(self.variable_dictionary_a[self.variables[i]]),
            )
        debug_print(logger, "Final - Initial Geometric Internal Coordinate Values:")
        for i in range(len(self.variables)):
            debug_print(
                logger,
                self.variables[i]
                + " = "
                + str(
                    self.variable_dictionary_a[self.variables[i]]
                    - self.variable_dictionary_b[self.variables[i]]
                ),
            )
        debug_print(logger, "Angstrom Bond Length Values:")
        for i in range(len(self.bond_indices)):
            debug_print(
                logger,
                str(self.index_dictionary[self.variables[i]])
                + " "
                + self.variables[i]
                + " = "
                + str(self.variable_dictionary_a[self.variables[i]] * self.Bohr_Ang),
            )
        if self.options.geom_check:
            Sum = 0
            bond_diff = []
            for i in range(len(self.bond_indices)):
                diff = (
                    self.variable_dictionary_a[self.variables[i]]
                    - self.variable_dictionary_b[self.variables[i]]
                )
                Sum += diff**2
                bond_diff.append(diff)
            ang_diff = []
            for i in range(len(self.variables) - len(self.bond_indices)):
                diff = (
                    self.variable_dictionary_a[
                        self.variables[i + len(self.bond_indices)]
                    ]
                    - self.variable_dictionary_b[
                        self.variables[i + len(self.bond_indices)]
                    ]
                )
                # Sum += diff**2
                ang_diff.append(diff)
            debug_print(logger, "max signed bond diff angstrom")
            debug_print(logger, np.max(bond_diff) * self.Bohr_Ang)
            debug_print(logger, "max signed bond diff bohr")
            debug_print(logger, np.max(bond_diff))
            debug_print(logger, "min signed bond diff angstrom")
            debug_print(logger, np.min(bond_diff) * self.Bohr_Ang)
            debug_print(logger, "min signed bond diff bohr")
            debug_print(logger, np.min(bond_diff))
            debug_print(logger, "max abs bond diff angstrom")
            debug_print(logger, np.max(np.abs(bond_diff)) * self.Bohr_Ang)
            debug_print(logger, "max abs bond diff bohr")
            debug_print(logger, np.max(np.abs(bond_diff)))
            debug_print(logger, bond_diff)
            debug_print(logger, "max signed ang diff")
            debug_print(logger, np.max(ang_diff))
            debug_print(logger, "min signed ang diff")
            debug_print(logger, np.min(ang_diff))
            debug_print(logger, "max abs ang diff")
            debug_print(logger, np.max(np.abs(ang_diff)))
            debug_print(logger, ang_diff)

            # debug_print(logger, "squared sum: ")
            # debug_print(logger, Sum)
            # debug_print(logger, "# of bonds:")
            # debug_print(logger, len(self.bond_indices))
            # debug_print(logger, "RMSD:")
            # debug_print(logger, np.sqrt(Sum / len(self.bond_indices)))

    def red_mass(self, indices):
        r = 0

        for i in indices:
            r += 1 / self.masses[int(i) - 1]

        return r

    def np_contains(self, array1, array2, tor=False):
        cont_bool = False
        for i in range(len(array1)):
            if np.array_equiv(array2, array1[i]):
                cont_bool = True
            if tor:
                if np.array_equiv(np.flip(array2), array1[i]):
                    cont_bool = True

        return cont_bool

    def _build_bonds(self, zmat_output):
        bond_count = 0
        if self.options.covalent_radii:
            c_r = CovalentRadii()
            indices = []
            transdisp_inter = TransfDisp(
                None,
                self,
                1,
                False,
                np.array([]),
                self.options,
                indices,
            )
            inter_atomic_len = np.zeros(
                (len(self.cartesians_b), len(self.cartesians_b))
            )
            N = len(self.cartesians_b)
            adj_mat = np.zeros((N, N))
            for i in range(len(self.cartesians_b)):
                for j in range(i):
                    inter_atomic_len[j, i] = transdisp_inter.calc_bond(
                        self.cartesians_b[i], self.cartesians_b[j]
                    )
                    if inter_atomic_len[j, i] < self.options.bond_threshold * (
                        c_r.get(self.atom_list[i]) + c_r.get(self.atom_list[j])
                    ):
                        bond_count += 1
                        adj_mat[i, j] = 1
                        adj_mat[j, i] = 1
                        self.bond_indices = np.append(self.bond_indices, 0)
                        self.bond_indices[-1] = np.array(
                            [str(j + 1), str(i + 1)], dtype=object
                        )
                        self.bond_variables.append("R" + str(bond_count))
            debug_print(logger, "Interatomic Distance Matrix:")
            debug_print(logger, inter_atomic_len)
            debug_print(logger, "Adjacency Matrix:")
            debug_print(logger, adj_mat)
            for i in range(len(adj_mat)):
                debug_print(logger, "Degree of vertex " + str(i))
                debug_print(logger, np.sum(adj_mat[i]))
            adj_mat2 = np.dot(adj_mat, adj_mat)
            for i in range(len(adj_mat2)):
                adj_mat2[i, i] = 0
            debug_print(logger, "Covalent Radius Scale Factor:")
            debug_print(logger, self.options.bond_threshold)
        else:
            for i in range(len(zmat_output)):
                if re.search(self.bond_regex, zmat_output[i]):
                    bond_count += 1
                    List = re.findall(self.bond_regex, zmat_output[i])[0]
                    self.bond_indices = np.append(self.bond_indices, 0)
                    self.bond_indices[-1] = np.array(List, dtype=object)
                    self.bond_variables.append("R" + str(bond_count))

    def _build_angles(self):
        ang_count = 0
        for i in range(len(self.bond_indices)):
            for j in range(len(self.bond_indices) - i - 1):
                a = np.setdiff1d(self.bond_indices[i], self.bond_indices[i + j + 1])
                b = np.intersect1d(self.bond_indices[i], self.bond_indices[i + j + 1])
                c = np.setdiff1d(self.bond_indices[i + j + 1], self.bond_indices[i])
                if len(a) and len(b) and len(c):
                    d = np.array([a[0], b[0], c[0]])
                    self.angle_indices = np.append(self.angle_indices, 0)
                    self.angle_indices[-1] = np.array(d, dtype=object)
                    ang_count += 1
                    self.angle_variables.append("A" + str(ang_count))

    def _build_four_center(self):
        for i in range(len(self.angle_indices)):
            for j in range(len(self.bond_indices)):
                a = np.setdiff1d(self.angle_indices[i], self.bond_indices[j])
                b = np.intersect1d(self.angle_indices[i], self.bond_indices[j])
                c = np.setdiff1d(self.bond_indices[j], self.angle_indices[i])
                if len(c) == 1:
                    d = np.where(self.bond_indices[j] == c)[0][0]
                    f = 1 - d
                    g = self.bond_indices[j][f]
                    h = np.where(self.angle_indices[i] == g)[0][0]
                    if h == 1:
                        # This is an out of plane bend
                        oop = np.array(
                            [
                                self.bond_indices[j][d],
                                self.bond_indices[j][f],
                                a[0],
                                a[1],
                            ],
                            dtype=object,
                        )

                        cont_bool = self.np_contains(self.oop_indices, oop)
                        if not cont_bool:
                            self.oop_indices = np.append(self.oop_indices, 0)
                            self.oop_indices[-1] = np.array(oop, dtype=object)
                    else:
                        # This is a torsion
                        if h:
                            tor = np.append(
                                self.angle_indices[i].copy(),
                                self.bond_indices[j][d],
                            )
                        else:
                            tor = np.append(
                                self.bond_indices[j][d],
                                self.angle_indices[i].copy(),
                            )
                        cont_bool = self.np_contains(
                            self.torsion_indices, tor, tor=True
                        )
                        if not cont_bool:
                            self.torsion_indices = np.append(self.torsion_indices, 0)
                            self.torsion_indices[-1] = np.array(tor, dtype=object)

        for i in range(len(self.torsion_indices)):
            self.torsion_variables.append("D" + str(i + 1))
        for i in range(len(self.oop_indices)):
            self.oop_variables.append("O" + str(i + 1))

    def _generate_topology(self):
        X_len_walks_dict = {}
        X_len_walks_dict["2_length_walks"] = self.bond_indices.copy()
        X_len_walks_dict["3_length_walks"] = self.angle_indices.copy()
        X_len_walks_dict["4_length_walks"] = self.torsion_indices.copy()

        count = 4

        prev_walks = self.torsion_indices.copy()

        while True:
            new_walks = np.array([])
            for i in range(len(prev_walks)):
                for j in range(len(self.bond_indices)):
                    a = np.where(prev_walks[i] == self.bond_indices[j][0])[0]
                    b = np.where(prev_walks[i] == self.bond_indices[j][1])[0]
                    if len(a) and not len(b):
                        if not a[0]:
                            new_walk = np.append(
                                self.bond_indices[j][1], prev_walks[i].copy()
                            )
                            new_walks = np.append(new_walks, new_walk)
                        elif a[0] == count - 1:
                            new_walk = np.append(
                                prev_walks[i].copy(), self.bond_indices[j][1]
                            )
                            new_walks = np.append(new_walks, new_walk)
                    if len(b) and not len(a):
                        if not b[0]:
                            new_walk = np.append(
                                self.bond_indices[j][0], prev_walks[i].copy()
                            )
                            new_walks = np.append(new_walks, new_walk)
                        elif b[0] == count - 1:
                            new_walk = np.append(
                                prev_walks[i].copy(), self.bond_indices[j][0]
                            )
                            new_walks = np.append(new_walks, new_walk)

            count += 1
            new_walks = new_walks.reshape((-1, count)).tolist()
            debug_print(logger, new_walks)
            new_walks = np.unique(new_walks, axis=0)

            del_list = np.array([])
            for i in range(len(new_walks)):
                for j in range(len(new_walks) - i - 1):
                    a = np.array([new_walks[i], np.flip(new_walks[i + j + 1])])
                    a = np.unique(a, axis=0)
                    if len(a) == 1:
                        del_list = np.append(del_list, [i + j + 1])

            del_list = del_list.astype(int)
            new_walks = np.delete(new_walks, del_list, axis=0)
            prev_walks = new_walks
            if count > self.options.topo_max_it or not len(new_walks):
                debug_print(
                    logger,
                    "Walk generator has terminated at walk lengths of " + str(count),
                )
                break
            X_len_walks_dict[str(count) + "_length_walks"] = new_walks
            debug_print(logger, str(count) + "_length_walks")
            debug_print(logger, len(new_walks))
            debug_print(logger, new_walks)

        dict_len = len(X_len_walks_dict) - 1

        debug_print(logger, "Looking for rings")
        cycles_dict = {}
        for i in range(dict_len):
            cycles_dict[str(i + 3)] = np.array([])
            for j in range(len(X_len_walks_dict[str(i + 3) + "_length_walks"])):
                a = np.array(
                    [
                        X_len_walks_dict[str(i + 3) + "_length_walks"][j][0],
                        X_len_walks_dict[str(i + 3) + "_length_walks"][j][-1],
                    ]
                )
                for k in range(len(self.bond_indices)):
                    b = np.intersect1d(a, self.bond_indices[k])
                    if len(b) == 2:
                        cycle = X_len_walks_dict[str(i + 3) + "_length_walks"][j].copy()
                        cycles_dict[str(i + 3)] = np.append(
                            cycles_dict[str(i + 3)], cycle
                        )
            cycles_dict[str(i + 3)] = cycles_dict[str(i + 3)].reshape((-1, i + 3))
            del_array = np.array([])
            if len(cycles_dict[str(i + 3)]):
                for j in range(len(cycles_dict[str(i + 3)])):
                    for k in range(len(cycles_dict[str(i + 3)]) - j - 1):
                        a = np.intersect1d(
                            cycles_dict[str(i + 3)][j],
                            cycles_dict[str(i + 3)][k + j + 1],
                        )
                        if len(a) == len(cycles_dict[str(i + 3)][0]):
                            del_array = np.append(del_array, [k + j + 1])
                del_array = del_array.astype(int)
                del_array = np.unique(del_array)
                cycles_dict[str(i + 3)] = np.delete(
                    cycles_dict[str(i + 3)], del_array, axis=0
                )
                debug_print(logger, f"{i+3}-membered ring detected!")
                debug_print(logger, cycles_dict[str(i + 3)])

    def _generate_walks(self):
        pass

    def _find_cycles(self):
        pass
