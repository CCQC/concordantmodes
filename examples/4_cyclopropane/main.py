from concordantmodes.options import Options

options_kwargs = {
    "cluster" : "sapelo",
    "program_init": "molpro",
    "program": "molpro",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_init": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_init": 7,
    "cart_insert": 7,
    "coords": "Custom",
    "success_regex": r"Molpro calculation terminated",
    "success_regex_init": r"Molpro calculation terminated",
    "symmetry" : True,
    "autosalcs" : True,
    #"gen_disps_init" : False,
    #"calc_init" : False,
    #"gen_disps" :False,
    #"calc" : False,
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
