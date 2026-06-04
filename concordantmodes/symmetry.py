from concordantmodes.masses import get_mass
import copy
from dataclasses import dataclass
from numpy import linalg as LA
import numpy as np
import scipy

try:
    import molsym
    from molsym.salcs.internal_coordinates import InternalCoordinates
    from molsym.salcs.cartesian_coordinates import CartesianCoordinates
    from molsym.salcs.projection_op import ProjectionOp

    print("imported")
except ImportError:
    raise ImportError("MolSym library not detected in your environment")


class Symmetry:

    def __init__(self, zmat, options, proj):
        self.zmat = zmat
        self.options = options
        self.proj = proj
        self.proj_irreps = None

    def dummy_obj(self):
        pass

    def run(self):
        schema = self.make_schema()
        mol = molsym.Molecule.from_schema(schema)

        self.symtext = molsym.Symtext.from_molecule(mol)
        if self.options.subgroup:
            print(f"The symmetry code is running in subgroup {self.options.subgroup}")
            self.symtext = self.symtext.subgroup_symtext(self.options.subgroup)
        print(self.options.second_order)
        if not self.options.second_order:
            print("internal coordinate salcs")
            self.molsym_salcs_ic()
        else:
            raise ValueError
            # this hasn't been implemented correctly yet. Its advisable to make it work within the C1 subgroup first...
            # self.molsym_salcs_cartesian()

    def molsym_salcs_cartesian(self):
        self.zmat.cartesians_b = self.symtext.mol.coords
        cart_coords = CartesianCoordinates(self.symtext)
        self.CDsalcs = ProjectionOp(self.symtext, cart_coords, project_Eckart=False)
        self.CDsalcs.sort_to("blocks")

    def molsym_salcs_ic(self):
        print(
            """
              Warning: A bug in MolSym has rendered the following code faulty if any
              type of linear coordinate is used. See https://github.com/NASymmetry/MolSym/issues/47
              """
        )
        zmat_coords = (
            self.zmat.bond_variables
            + self.zmat.angle_variables
            + self.zmat.torsion_variables
            + self.zmat.oop_variables
            + self.zmat.linx_variables
            + self.zmat.liny_variables
        )
        self.ic_list = []
        for var in self.zmat.variables:
            new_coord = self.remove_one(self.zmat.index_dictionary[var])
            self.ic_list.append(new_coord)

        ics = []
        for i in range(len(self.ic_list)):
            ics.append([self.ic_list[i], zmat_coords[i]])
        print(f"The Internal Coordinates for MolSym {ics}")
        if not self.options.man_proj:
            ICs = InternalCoordinates(self.symtext, ics)
            self.salcs = ProjectionOp(self.symtext, ICs)
            self.salcs.sort_to("blocks")
            self.nbfxns = len(ics)

            self.package_salcs()

        else:
            """
            Make your own salcs via manual projection matrix, which is passed into
            the CMA code as "proj." The code below will pass it into MolSym's projection
            operator and it will return a sym_sort for it.
            """
            ICs = InternalCoordinates(self.symtext, ics)
            self.salcs, list_salc_ids = ProjectionOp(
                self.symtext, ICs, False, self.proj.T
            )
            sym_sort = []
            for ir, irrep in enumerate(self.symtext.irreps):
                if irrep.symbol in list_salc_ids:
                    sym_sort.append(
                        [i for i, x in enumerate(list_salc_ids) if x == irrep.symbol]
                    )
            self.sym_sort = sym_sort

    def make_schema(self):
        qc_obj = {
            "symbols": self.zmat.atom_list,
            "mass": [get_mass(x) for x in self.zmat.atom_list],
            "geometry": self.zmat.cartesians_b.flatten().tolist(),
        }
        return qc_obj

    def remove_one(self, coordinate_index):
        new_indices = []
        for index in coordinate_index:
            new = int(index) - 1
            new_indices.append(new)
        return new_indices

    def package_salcs(self):
        self.irreps = [
            self.salcs.irreps[i].symbol for i in range(len(self.salcs.irreps))
        ]
        sdict = dict.fromkeys(self.irreps)
        fxn_list = []
        self.salcs.salc_sets = []
        for ir, irrep in enumerate(self.symtext.irreps):
            sdict[irrep.symbol] = {}
            if len(self.salcs.salcs_by_irrep[ir]) == 0:
                ir_salcs = np.zeros((0, self.nbfxns))
                fxn_list.append([])
                self.salcs.salc_sets.append(ir_salcs)
            else:
                ir_salcs = [self.salcs[i].coeffs for i in self.salcs.salcs_by_irrep[ir]]
                ir_salcs = np.row_stack(ir_salcs)
                self.salcs.salc_sets.append(ir_salcs)
                fxn_list.append(
                    [1 for i in range(0, (len(ir_salcs) // self.symtext.irreps[ir].d))]
                )

        self.irreplength = []
        if self.options.exploit_degen and not self.options.partner_functions:
            for s, SET in enumerate(self.salcs.salc_sets):
                degen = self.symtext.irreps[s].d
                if SET.shape[0] == 0:
                    self.irreplength.append(0)
                else:
                    self.irreplength.append(SET.shape[0] // degen)
        else:
            for SET in self.salcs.salc_sets:
                self.irreplength.append(SET.shape[0])

    def remove_redun(self):
        np.set_printoptions(threshold=np.inf, precision=8)
        Proj = []
        tol = 1e-5
        proj_irreps = []
        self.salcblocks = []
        for h, block in enumerate(self.b):
            proj, eigs, _ = LA.svd(block)
            proj[np.abs(proj) < tol] = 0
            proj_array = np.array(np.where(np.abs(eigs) > tol))
            newproj = proj.T[: len(proj_array[0])]
            newproj = newproj.T
            proj_irreps.append(newproj.shape[1])
            Proj.append(newproj)
            self.salcblocks.append(
                np.dot(newproj.T, self.salcs.salc_sets[h][: self.irreplength[h]])
            )

        blocked_proj = []
        for pi, p in enumerate(Proj):
            if p is None:
                continue
            else:
                blocked_proj = scipy.linalg.block_diag(blocked_proj, p)
        bloc = np.delete(blocked_proj, 0, 0)
        return bloc, proj_irreps

    def make_proj(self, s_vec):
        np.set_printoptions(threshold=np.inf, precision=10)
        print("Project out redundant coordinates to make the nonredundant set")
        self.b = []
        for h, s in enumerate(self.salcs.salc_sets):
            self.b.append(np.dot(s[: self.irreplength[h]], s_vec.B))
        self.new_proj, self.proj_irreps = self.remove_redun()
        s_vec.proj = copy.deepcopy(self.new_proj)
        sblock = []
        self.symtext.salcblocks = copy.deepcopy(self.salcblocks)
        for s, salc in enumerate(self.salcblocks):
            if salc is None:
                continue
            elif len(salc) == 0:
                continue
            else:
                sblock.append(salc)
        newsblock = []
        for s in sblock:
            newsblock.append(s)
        self.sblock = copy.deepcopy(sblock)

        self.salc_proj = np.row_stack(newsblock).T

    def create_flat_sym_sort(self, sym_sort):
        print("sym_sort: ")
        print(sym_sort)
        self.flat_sym_sort = np.array([])
        self.flat_sym_sort_inv = np.array([])
        for i in range(len(sym_sort)):
            self.flat_sym_sort = np.append(self.flat_sym_sort, sym_sort[i])

        self.flat_sym_sort = self.flat_sym_sort.astype(int)

        print("Flattened sym_sort: ")
        print(self.flat_sym_sort)
        for i in range(len(self.flat_sym_sort)):
            self.flat_sym_sort_inv = np.append(
                self.flat_sym_sort_inv, np.where(self.flat_sym_sort == i)[0][0]
            )
        # flat_sym_sort_inv = flat_sym_sort_inv[flat_sym_sort]
        self.flat_sym_sort_inv = self.flat_sym_sort_inv.astype(int)

    def GF_sym_sort(self, F, g_mat, sym_sort):
        Fbuff1 = np.array([])
        Fbuff2 = {}
        Gbuff1 = np.array([])
        Gbuff2 = {}
        for i in range(len(sym_sort)):
            Fbuff1 = F.copy()
            Fbuff1 = Fbuff1[sym_sort[i]]
            Fbuff1 = np.array([Fbuff1[:, sym_sort[i]]])
            Fbuff2[str(i)] = Fbuff1.copy()
            Gbuff1 = g_mat.G.copy()
            Gbuff1 = Gbuff1[sym_sort[i]]
            Gbuff1 = np.array([Gbuff1[:, sym_sort[i]]])
            Gbuff2[str(i)] = Gbuff1.copy()
        Fbuff3 = Fbuff2[str(0)][0].copy()
        Gbuff3 = Gbuff2[str(0)][0].copy()
        for i in range(len(sym_sort) - 1):
            Fbuff3 = np.block(
                [
                    [Fbuff3, np.zeros((len(Fbuff3), len(Fbuff2[str(i + 1)][0])))],
                    [
                        np.zeros((len(Fbuff2[str(i + 1)][0]), len(Fbuff3))),
                        Fbuff2[str(i + 1)][0],
                    ],
                ]
            )
            Gbuff3 = np.block(
                [
                    [Gbuff3, np.zeros((len(Gbuff3), len(Gbuff2[str(i + 1)][0])))],
                    [
                        np.zeros((len(Gbuff2[str(i + 1)][0]), len(Gbuff3))),
                        Gbuff2[str(i + 1)][0],
                    ],
                ]
            )
        F = Fbuff3[self.flat_sym_sort_inv]
        F = F[:, self.flat_sym_sort_inv]
        g_mat.G = Gbuff3[self.flat_sym_sort_inv]
        g_mat.G = g_mat.G[:, self.flat_sym_sort_inv]

        F_sym = F[self.flat_sym_sort].copy()
        F_sym = F_sym[:, self.flat_sym_sort]
        print("Sym Force Constants:")
        print(F_sym)

        g_sym = g_mat.G[self.flat_sym_sort].copy()
        g_sym = g_sym[:, self.flat_sym_sort]
        g_sym[np.abs(g_sym) < 1e-9] = 0
        print("Sym G-Matrix:")
        print(sym_sort)
        print(g_sym)
        return F, g_mat.G

    def cma2_sym_sort(self, sym_sort, od_inds, irreps_b, F_inter, xi, xi_tol_i):
        total_off_diags_buff = 0
        for irrep in irreps_b:
            if len(irrep) > 1:
                for i in range(len(irrep)):
                    for j in range(i):
                        if i != j:
                            a = irrep[i]
                            b = irrep[j]
                            buff = np.abs(F_inter[a, b])
                            xi[a, b] = buff / np.sqrt(
                                np.abs(F_inter[a, a]) * np.abs(F_inter[b, b])
                            )
                            if xi[a, b] > xi_tol_i:
                                od_inds.append([a, b])
                total_off_diags_buff += (len(irrep) ** 2 - len(irrep)) / 2
        return od_inds, total_off_diags_buff

    def create_sym_sort_disps(self, sym_sort, indices):
        sym_disps = []
        for i in sym_sort:
            for j in indices:
                if j[0] in i and j[1] in i:
                    sym_disps.append([j[0], j[1]])
        return sym_disps

    def mode_symmetry_sort(self, TED, sym_sort, freqs):
        ref_TED_b = TED
        sym_modes = []
        for irrep in sym_sort:
            irrep_modes = []
            for i in range(len(ref_TED_b.T)):
                Sum = 0
                for j in irrep:
                    Sum += ref_TED_b.T[i, j]
                if Sum > 80.0:
                    irrep_modes.append(i)
            if len(irrep_modes) != len(irrep):
                print("Something's wrong with the irrep symmetry sorter:")
                raise RuntimeError
            sym_modes.append(irrep_modes)

        sym_freqs = copy.deepcopy(sym_modes)
        del_list = []
        for i in range(len(sym_modes)):
            for j in range(len(sym_modes[i])):
                index = sym_modes[i][j]
                sym_freqs[i][j] = freqs[index].copy()
            sym_freqs[i].reverse()
        del_list.reverse()
        if len(del_list):
            for i in del_list:
                print(freqs[sym_modes[i][0]])
        flat_sym_freqs = [x for xs in sym_freqs for x in xs]
        flat_sym_freqs = np.array(flat_sym_freqs)

        return sym_modes, flat_sym_freqs
