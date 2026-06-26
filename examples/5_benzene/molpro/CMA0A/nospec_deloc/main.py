from concordantmodes.options import Options

options_kwargs = {
    "cluster": "slurm",
    "program_b": "molpro",
    "program_a": "molpro",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "energy_regex_b": r"!RHF STATE 1.1 Energy\s*\s+(\-\d+\.\d+)",
    "cart_insert_b": 7,
    "cart_insert_a": 7,
    "covalent_radii": True,
    "coords": "Delocalized",
    "topo_analysis": True,
    "calc_b" : False,
    # "calc_a" : False,
    "gen_disps_b" : False,
    # "gen_disps_a" : False,
    "success_regex_a": r"Molpro calculation terminated",
    "success_regex_b": r"Molpro calculation terminated",
    # "symmetry": True,
    # "autosalcs": True,
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
