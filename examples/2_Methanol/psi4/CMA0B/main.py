from concordantmodes.options import Options

options_kwargs = {
    "cluster": "sapelo",
    "program": "psi4",
    "energy_regex": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert": 7,
    "coords": "Delocalized",
    "success_regex": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
