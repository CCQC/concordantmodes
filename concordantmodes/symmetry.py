from concordantmodes.masses import get_mass
import copy
from dataclasses import dataclass
from numpy import linalg as LA
import numpy as np
import scipy


class Symmetry(object):
    
    def __init__(self, zmat, options, proj):
        self.zmat = zmat
        self.options = options
        self.proj = proj
    
    def dummy_obj(self):
        pass

    def run(self):
        try:
            import molsym
            from molsym.salcs.internal_coordinates import InternalCoordinates
            from molsym.salcs.projection_op import ProjectionOp
        except ImportError:
            raise ImportError('MolSym library not detected in your environment')
        zmat_coords = self.zmat.bond_variables + self.zmat.angle_variables + self.zmat.torsion_variables + self.zmat.oop_variables + self.zmat.linx_variables + self.zmat.liny_variables
        schema = self.make_schema()
        self.ic_list = []
        for var in self.zmat.variables:
            new_coord = self.remove_one(self.zmat.index_dictionary[var])
            self.ic_list.append(new_coord)


        ics = []
        for i in range(len(self.ic_list)):
            ics.append([self.ic_list[i], zmat_coords[i]])
        print(f"The Internal Coordinates for MolSym {ics}")
        mol = molsym.Molecule.from_schema(schema)
        self.symtext = molsym.Symtext.from_molecule(mol)
        if not self.options.man_proj:
            ICs = InternalCoordinates(self.symtext, ics)
            self.salcs = ProjectionOp(self.symtext, ICs)
            self.salcs.sort_to("blocks")
            self.nbfxns = len(ics)

            self.package_salcs()
            #print(self.salcs)
            #print(vars(self.salcs))
            #print(self.salcs.salc_sets[2])
            
            #self.salcs, list_salc_ids = ProjectionOp(self.symtext, ICs, False, self.salcs.salc_sets[2])
        else:
            print("Make your own salcs via manual projection matrix")
            ICs = InternalCoordinates(self.symtext, ics)
            self.salcs, list_salc_ids = ProjectionOp(self.symtext, ICs, False, self.proj.T)
            sym_sort = []
            for ir, irrep in enumerate(self.symtext.irreps):
                if irrep.symbol in list_salc_ids:
                    sym_sort.append([i for i, x in enumerate(list_salc_ids) if x == irrep.symbol])
            self.sym_sort = sym_sort 

    def make_schema(self):
        qc_obj = {
            "symbols": self.zmat.atom_list,
            "mass" : [get_mass(x) for x in self.zmat.atom_list],
            "geometry" : self.zmat.cartesians_init.flatten().tolist(),
        }
        return qc_obj

    def remove_one(self, coordinate_index):
        new_indices = []
        for index in coordinate_index:
            new = int(index) - 1
            new_indices.append(new)
        return new_indices

    def package_salcs(self):
        self.irreps = [self.salcs.irreps[i].symbol for i in range(len(self.salcs.irreps))]
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
                fxn_list.append([1 for i in range(0, (len(ir_salcs) // self.symtext.irreps[ir].d))])

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
            self.salcblocks.append(np.dot(newproj.T, self.salcs.salc_sets[h][:self.irreplength[h]]))

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
        print("Project out redundent coordinates to make the nonredundant set")
        self.b = []
        for h, s in enumerate(self.salcs.salc_sets):
            self.b.append(np.dot(s[:self.irreplength[h]], s_vec.B))
        self.new_proj, self.proj_irreps = self.remove_redun()
        s_vec.proj = copy.deepcopy(self.new_proj)
        sblock = []
        self.symtext.salcblocks = copy.deepcopy(self.salcblocks)
        for s, salc in enumerate(self.salcblocks):
            if salc is None:
                continue
            elif  len(salc) == 0:
                continue
            else:
                sblock.append(salc)
        newsblock = []
        for s in sblock:
            newsblock.append(s)
        self.sblock = copy.deepcopy(sblock)

        self.salc_proj = np.row_stack(newsblock).T