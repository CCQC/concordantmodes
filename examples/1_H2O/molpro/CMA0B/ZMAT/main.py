from concordantmodes.options import Options

options_kwargs = {
    "program_a": "molpro",
    "cluster" : "sapelo",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_a": 9,
    "success_regex_a": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
