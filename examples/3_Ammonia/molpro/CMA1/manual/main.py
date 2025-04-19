from concordantmodes.options import Options
from scipy.linalg import block_diag
from numpy.linalg import norm
import numpy as np
options_kwargs = {
    "queue": "gen4.q,gen6.q,debug.q",
    "program_init": "molpro@2010.1.67+mpi",
    "program": "molpro@2010.1.67+mpi",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_init": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_init": 7,
    "cart_insert": 7,
    "coords": "Custom",
    "success_regex": r"Variable memory released",
    "success_regex_init": r"Variable memory released",
    #"symmetry" : False,
    "symmetry" : True,
    #"man_proj" : True,
    "calc_init" :False,
    "gen_disps_init" :False,
    "calc" : False,
    "gen_disps": False,
}
options_obj = Options(**options_kwargs)

stretches_mat = np.array([
        [(1/np.sqrt(3)), ( 1/np.sqrt(3)), (1/np.sqrt(3))],
        [(2/np.sqrt(6)), (-1/np.sqrt(6)), (-1/np.sqrt(6))],
        [0             , (1/np.sqrt(2)), (-1/np.sqrt(2))],
        ]).T
angles = np.array([
        [(2/np.sqrt(6)), (-1/np.sqrt(6)), (-1/np.sqrt(6))],
        [0,              (1/np.sqrt(2)),  (-1/np.sqrt(2))],
        ])
angles = angles.T
oop_mat = np.array([
        [(1/np.sqrt(3)), (1/np.sqrt(3)), (1/np.sqrt(3))],
        ])
oop_mat = oop_mat.T
        
#raise RuntimeError
        
Proj = block_diag(stretches_mat,angles,oop_mat)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj, Proj)
CMA_obj.run()
