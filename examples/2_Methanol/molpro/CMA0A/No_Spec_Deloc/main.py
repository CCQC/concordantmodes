from concordantmodes.options import Options

options_kwargs = {
    "cluster": "slurm",
    "program_b": "molpro",
    "program_a": "molpro",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_b": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_b": 9,
    "cart_insert_a": 9,
    "covalent_radii": True,
    "coords": "Delocalized",
    "topo_analysis": True,
    # "calc_b" : False,
    # "calc_a" : False,
    # "gen_disps_b" : False,
    # "gen_disps_a" : False,
    "success_regex_b": r"Molpro calculation terminated",
    "success_regex_a": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
