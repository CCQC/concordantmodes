from concordantmodes.options import Options

options_kwargs = {
    "cluster": "slurm",
    "program_a": "psi4",
    "energy_regex_a": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert_a": 7,
    "success_regex_a": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
