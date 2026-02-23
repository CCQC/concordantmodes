from concordantmodes.options import Options

options_kwargs = {
    "cluster": "slurm",
    "program_b": "psi4",
    "program_a": "psi4",
    "deriv_level_b": 1,
    "energy_regex_a": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "energy_regex_b": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "gradient_regex_b": [r"Total Gradient", r"tstop"],
    "cart_insert_b": 7,
    "cart_insert_a": 7,
    "gen_disps_b": False,
    "calc_b": False,
    "coords": "Delocalized",
    "success_regex_b": r"beer",
    "success_regex_a": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
