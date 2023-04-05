from concordantmodes.options import Options

options_kwargs = {
    "queue": "gen3.q,gen4.q,gen6.q,debug.q",
    "program_init": "psi4@master",
    "program": "psi4@master",
    "energy_regex": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "energy_regex_init": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert_init": 7,
    "cart_insert": 7,
    "calc_init": False,
    "calc": False,
    "coords": "Redundant",
    "success_regex_init": r"beer",
    "success_regex": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
