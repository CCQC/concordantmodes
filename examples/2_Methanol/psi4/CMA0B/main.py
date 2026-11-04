from concordantmodes.options import Options

options_kwargs = {
    # CMA-0B: the level-B Hessian is read from force_constants/fc_cart_b.dat (Cartesian,
    # CFOUR FCMFINAL format) instead of being computed.
    "cart_fc_b": True,
    "cluster": "slurm",
    "program_a": "psi4",
    "energy_regex_a": r"Giraffe The Energy is\s+(\-\d+\.\d+)",
    "cart_insert_a": 7,
    "coords": "Delocalized",
    "success_regex_a": r"beer",
}
options_obj = Options(**options_kwargs)

# 3. call Concordant Modes Program
from concordantmodes.cma import ConcordantModes

CMA_obj = ConcordantModes(options_obj)
CMA_obj.run()
