from concordantmodes.options import Options

options_kwargs = {
    "cluster" : "sapelo",
    "program_a": "psi4",
    "program_b": "psi4",
    "energy_regex_a": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "energy_regex_b": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert_a": 7,
    "cart_insert_b": 7,
    "coords": "Delocalized",
    "success_regex_a": r"beer",
    "success_regex_b": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
