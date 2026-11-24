from concordantmodes.options import Options

options_kwargs = {
    # CMA-0B: the level-B Hessian is read from force_constants/fc_cart_b.dat (Cartesian,
    # CFOUR FCMFINAL format) instead of being computed.
    "cart_fc_b": True,
    "program_a": "molpro",
    "cluster": "slurm",
    "energy_regex_a": r"\(T\) total energy\s+(\-\d+\.\d+)",
    "cart_insert_a": 9,
    "success_regex_a": r"Molpro calculation terminated",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
