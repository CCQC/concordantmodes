from concordantmodes.options import Options

options_kwargs = {
    "cluster" : "sapelo",
    "program": "molpro",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert": 9,
    "coords" : "Delocalized",
    "success_regex": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
