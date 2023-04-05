from concordantmodes.options import Options

options_kwargs = {
    "queue": "gen4.q,gen6.q",
    "program": "molpro@2010.1.67+mpi",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert": 9,
    "coords": "Redundant",
    "success_regex": r"Variable memory released",
    "calc": True,
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
