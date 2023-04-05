from concordantmodes.options import Options

options_kwargs = {
    "queue": "gen4.q,gen6.q,debug.q",
    "program_init": "molpro@2010.1.67+mpi",
    "program": "molpro@2010.1.67+mpi",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_init": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_init": 9,
    "cart_insert": 9,
    "coords": "Redundant",
    "success_regex": r"Variable memory released",
    "success_regex_init": r"Variable memory released",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
