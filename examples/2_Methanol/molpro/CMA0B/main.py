from concordantmodes.options import Options

options_kwargs = {
    "cluster": "slurm",
    "program_a": "molpro",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_a": 9,
    "coords": "Delocalized",
    "success_regex_a": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
