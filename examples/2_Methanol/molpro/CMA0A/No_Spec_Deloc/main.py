from concordantmodes.options import Options

options_kwargs = {
    "cluster" : "sapelo",
    "program_init": "molpro",
    "program": "molpro",
    "energy_regex": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_init": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_init": 9,
    "cart_insert": 9,
    # "calc_init" : False,
    # "gen_disps_init" : False,
    # "reduced_disp" : True,
    "covalent_radii": True,
    "coords": "Delocalized",
    "success_regex": r"Molpro calculation terminated",
    "success_regex_init": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
