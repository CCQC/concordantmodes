from concordantmodes.options import Options

options_kwargs = {
    "queue": "gen3.q,gen4.q,gen6.q,debug.q",
    "program": "psi4@master",
    "energy_regex": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert": 7,
    "coords": "Redundant",
    "success_regex": r"beer",
    "interatomic_distance": True,
    "inter_tol": 2.0,
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
