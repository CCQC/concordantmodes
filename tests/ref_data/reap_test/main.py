from concordantmodes.options import Options

options_kwargs = {
    "queue": "gen3.q,gen4.q,gen6.q,debug.q",
    "program": "psi4@master",
    "energy_regex": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert": 7,
    "coords": "Redundant",
    "success_regex": r"beer",
    "calc": False,
    "gen_disps": False,
    "off_diag": False,
    "off_diag_bands": 3,
    "off_diag_limit": False,
    "printout_rel_e": False,
    "mode_coupling_check": False,
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
