"""User options for a Concordant Modes run.

Every option is declared once in ``OPTION_SPECS`` with its default, the
values it accepts, and a one-line description. That table drives:

* validation - unknown names raise an error with "did you mean" hints,
  and values are checked when ``Options`` is created and again when a run
  starts;
* cluster profiles (``cluster_profile``) and the per-program parsing
  presets (``program_*`` / ``method_*``, see :mod:`concordantmodes.programs`);
* the options reference in the documentation and ``cma options``.

Options are plain attributes, so ``options.coords = "Custom"`` still works,
but assigning a name that is not an option raises an error.
"""

import difflib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from concordantmodes import programs
from concordantmodes.errors import OptionsError


# ----------------------------------------------------------------------
# Option table
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OptionSpec:
    name: str
    default: Any
    group: str
    help: str
    choices: Optional[Tuple[str, ...]] = None  # case-insensitive strings
    types: Optional[Tuple[type, ...]] = None
    check: Optional[Callable[[Any], Optional[str]]] = None
    no_effect: bool = False  # accepted for old inputs but currently unused
    internal: bool = False  # set by the program itself; not documented
    allow_none: bool = False


def _positive(v):
    return None if v > 0 else "must be greater than 0"


def _non_negative(v):
    return None if v >= 0 else "must be 0 or greater"


def _deriv_level(v):
    return None if v in (0, 1) else "must be 0 (energies) or 1 (gradients)"


def _regex_pair(v):
    if v in ("", None):
        return None
    if (
        isinstance(v, (list, tuple))
        and len(v) == 2
        and all(isinstance(x, str) for x in v)
    ):
        return None
    return "must be a list of two regular expressions [start, end]"


def _str_list(v):
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return None
    return "must be a list of strings"


B, I, F, S = (bool,), (int,), (int, float), (str,)

G_INPUT = "Input and geometry"
G_COORD = "Internal coordinates"
G_LEVEL = "Levels of theory and output parsing"
G_DISP = "Displacements and force constants"
G_JOBS = "Running the calculations"
G_SYM = "Symmetry"
G_OUT = "Output and analysis"
G_ADV = "Advanced"

GROUP_ORDER = (G_INPUT, G_COORD, G_LEVEL, G_DISP, G_JOBS, G_SYM, G_OUT, G_ADV)

# fmt: off
_SPECS = [
    # --- Input and geometry -------------------------------------------------
    OptionSpec("cart_coords", "Bohr", G_INPUT, "Units of the Cartesian geometry in the zmat file.", choices=("Bohr", "Angstrom")),
    OptionSpec("cart_insert_a", -1, G_INPUT, "Line number in templateA.dat after which the geometry is inserted. Leave at -1 and put a {geometry} line in the template instead.", types=I),
    OptionSpec("cart_insert_b", -1, G_INPUT, "Same as cart_insert_a, for templateB.dat.", types=I),
    OptionSpec("cart_insert_c", -1, G_INPUT, "Same as cart_insert_a, for templateC.dat (CMA-2, experimental).", types=I),
    OptionSpec("geom_check", False, G_INPUT, "Read the zmat file, report the structure and coordinates, and stop.", types=B),
    # --- Internal coordinates ----------------------------------------------
    OptionSpec("coords", "ZMAT", G_COORD, "Internal coordinate system: 'ZMAT' (from the Z-matrix), 'Custom' (the coordinates listed in the zmat file) or 'Delocalized' (all bonds/angles/torsions, made non-redundant).", choices=("ZMAT", "Custom", "Delocalized")),
    OptionSpec("covalent_radii", False, G_COORD, "Find bonds automatically from covalent radii instead of the zmat connectivity.", types=B),
    OptionSpec("bond_threshold", 1.2, G_COORD, "With covalent_radii, atoms closer than bond_threshold x (sum of covalent radii) are bonded.", types=F, check=_positive),
    OptionSpec("topo_analysis", False, G_COORD, "With Delocalized coordinates, run the ring/topology analysis on the bond graph.", types=B),
    OptionSpec("topo_max_it", 20, G_COORD, "Longest bond walk (ring size) considered by topo_analysis.", types=I, check=_positive),
    OptionSpec("man_proj", False, G_COORD, "Use a projection matrix you supply (ConcordantModes(options, proj=...)) instead of building one.", types=B),
    OptionSpec("red_tol", 1.0e-4, G_COORD, "Singular values of the B-matrix below this are treated as redundant coordinates.", types=F, check=_positive),
    # --- Levels of theory and parsing -------------------------------------
    OptionSpec("program_a", "molpro", G_LEVEL, "Program for the high-level (A) single points: molpro, psi4, orca or cfour (an '@version' suffix is allowed).", types=S),
    OptionSpec("program_b", "molpro", G_LEVEL, "Program for the low-level (B) calculations.", types=S),
    OptionSpec("program_c", "molpro", G_LEVEL, "Program for level C (CMA-2, experimental).", types=S),
    OptionSpec("method_a", "", G_LEVEL, "Method whose energy is read from level-A outputs, e.g. 'ccsd(t)'. Only needed for Molpro inputs that run several methods; empty uses the program's final energy.", types=S),
    OptionSpec("method_b", "", G_LEVEL, "Same as method_a, for level B.", types=S),
    OptionSpec("method_c", "", G_LEVEL, "Same as method_a, for level C.", types=S),
    OptionSpec("energy_regex_a", "", G_LEVEL, "Regular expression with one group capturing the level-A energy. Empty uses the built-in preset for program_a/method_a.", types=S),
    OptionSpec("energy_regex_b", "", G_LEVEL, "Same as energy_regex_a, for level B.", types=S),
    OptionSpec("energy_regex_c", "", G_LEVEL, "Same as energy_regex_a, for level C.", types=S),
    OptionSpec("success_regex_a", "", G_LEVEL, "Regular expression that must appear in a finished level-A output. Empty uses the program's normal-termination line.", types=S),
    OptionSpec("success_regex_b", "", G_LEVEL, "Same as success_regex_a, for level B.", types=S),
    OptionSpec("success_regex_c", "", G_LEVEL, "Same as success_regex_a, for level C.", types=S),
    OptionSpec("gradient_regex_a", "", G_LEVEL, "[start, end] regular expressions bracketing the gradient block (deriv_level_a = 1). Empty uses the program preset.", check=_regex_pair),
    OptionSpec("gradient_regex_b", "", G_LEVEL, "Same as gradient_regex_a, for level B.", check=_regex_pair),
    OptionSpec("gradient_regex_c", "", G_LEVEL, "Same as gradient_regex_a, for level C.", check=_regex_pair),
    # --- Displacements and force constants ---------------------------------
    OptionSpec("deriv_level_a", 0, G_DISP, "0: level-A force constants from energies; 1: from gradients.", types=I, check=_deriv_level),
    OptionSpec("deriv_level_b", 0, G_DISP, "0: level-B force constants from energies; 1: from gradients.", types=I, check=_deriv_level),
    OptionSpec("deriv_level_c", 0, G_DISP, "Same as deriv_level_a, for level C.", types=I, check=_deriv_level),
    OptionSpec("disp_a", 0.01, G_DISP, "Displacement size for level A (normal-coordinate units).", types=F, check=_positive),
    OptionSpec("disp_b", 0.01, G_DISP, "Displacement size for level B (internal-coordinate units).", types=F, check=_positive),
    OptionSpec("disp_c", 0.01, G_DISP, "Displacement size for level C.", types=F, check=_positive),
    OptionSpec("off_diag", 0, G_DISP, "Off-diagonal level-A force constants: 0 = diagonal only (CMA-0); 2 = CMA-2 couplings chosen by xi_tol (experimental).", types=I),
    OptionSpec("xi_tol", 0.02, G_DISP, "CMA-2: couple modes i,j when |F_ij|/sqrt(|F_ii F_jj|) exceeds this.", types=F, check=_non_negative),
    OptionSpec("de_novo_C", False, G_DISP, "CMA-2: compute level C from new calculations instead of reusing level-B outputs.", types=B),
    OptionSpec("reduced_disp", False, G_DISP, "Adapt the level-A step sizes to each mode's curvature using the level-B force constants (see reduced_disp_size).", types=B),
    OptionSpec("reduced_disp_size", 0.0025, G_DISP, "Scale s in the reduced_disp step size Delta_i = s / |F_ii|^(1/4).", types=F, check=_positive),
    OptionSpec("scaled_disp", False, G_DISP, "Divide each level-B displacement by the largest component of its projection vector, so the most-displaced primitive coordinate moves by disp_b.", types=B),
    OptionSpec("tight_disp", False, G_DISP, "Converge the internal-to-Cartesian back-transformation more tightly.", types=B),
    OptionSpec("transf_max_it", 50, G_DISP, "Maximum iterations of the internal-to-Cartesian back-transformation.", types=I, check=_positive),
    OptionSpec("transf_tol", 1.0e-7, G_DISP, "Convergence threshold of the back-transformation.", types=F, check=_positive),
    OptionSpec("second_order", False, G_DISP, "Include second-order (B2) terms when converting force constants; needed away from stationary points.", types=B),
    OptionSpec("cart_fc_b", False, G_DISP, "Compute the level-B Hessian in Cartesian instead of internal coordinates.", types=B),
    OptionSpec("conv_grad", True, G_DISP, "Convert Cartesian gradients to internal coordinates when reading them.", types=B),
    # --- Running the calculations ------------------------------------------
    OptionSpec("cluster", "manual", G_JOBS, "'manual': CMA writes the inputs and you run them, then rerun CMA. 'slurm' or 'sge': CMA submits, waits and continues. 'custom': CMA runs custom_submit_str in each displacement directory.", choices=("manual", "slurm", "sge", "custom")),
    OptionSpec("cluster_profile", "", G_JOBS, "Named group of cluster settings applied before your own options, e.g. 'sapelo2_hfslab'. See the profiles section of the docs.", types=S),
    OptionSpec("job_template", "", G_JOBS, "Path to your own job script. Placeholders such as {nslots}, {memory}, {time_limit}, {queue}, {job_name}, {run_command}, {job_setup} and {directives} are filled in.", types=S),
    OptionSpec("job_name", "Concordant", G_JOBS, "Scheduler job name.", types=S),
    OptionSpec("queue", "batch", G_JOBS, "Slurm partition or SGE queue.", types=S),
    OptionSpec("nslots", 1, G_JOBS, "CPU cores per displacement calculation ($NSLOTS in the job script).", types=I, check=_positive),
    OptionSpec("memory", "32GB", G_JOBS, "Memory per displacement job, in scheduler syntax (e.g. '32GB').", types=S),
    OptionSpec("time_limit", "04:00:00", G_JOBS, "Wall-time limit per displacement job.", types=S),
    OptionSpec("scheduler_directives", [], G_JOBS, "Extra scheduler lines added to the job script, e.g. ['#SBATCH --constraint=EPYC'].", check=_str_list),
    OptionSpec("job_setup", "", G_JOBS, "Shell lines run before the program in each job (module load, conda activate, scratch setup...).", types=S),
    OptionSpec("run_command_a", "", G_JOBS, "Command that runs one level-A calculation in its directory. Empty uses the program's default command.", types=S),
    OptionSpec("run_command_b", "", G_JOBS, "Same as run_command_a, for level B.", types=S),
    OptionSpec("run_command_c", "", G_JOBS, "Same as run_command_a, for level C.", types=S),
    OptionSpec("custom_submit_str", "", G_JOBS, "With cluster='custom': command run in each displacement directory, followed by the path of sub_script.sh.", types=S),
    OptionSpec("poll_interval", 30, G_JOBS, "Seconds between scheduler status checks.", types=F, check=_positive),
    OptionSpec("slurm_array", True, G_JOBS, "Slurm: submit displacements as job arrays (False: one job per displacement).", types=B),
    OptionSpec("max_array_size", 5000, G_JOBS, "Largest job array to submit; bigger runs are split. Match your cluster's MaxArraySize.", types=I, check=_positive),
    OptionSpec("array_throttle", None, G_JOBS, "Maximum array tasks running at once (Slurm '%N'); None for no limit.", types=I, check=_positive, allow_none=True),
    # --- Symmetry -------------------------------------------------------------
    OptionSpec("molsym_symmetry", False, G_SYM, "Use MolSym to build symmetry-adapted coordinates and skip symmetry-equivalent displacements.", types=B),
    OptionSpec("subgroup", False, G_SYM, "With molsym_symmetry, work in this point subgroup (e.g. 'C2v').", types=(bool, str)),
    OptionSpec("exploit_degen", True, G_SYM, "Use degeneracies to skip partner-function displacements.", types=B),
    OptionSpec("partner_functions", True, G_SYM, "Keep both partner functions of degenerate irreps.", types=B),
    OptionSpec("exploit_pm_symm", False, G_SYM, "Skip minus displacements that are symmetry-equivalent to plus displacements.", types=B),
    OptionSpec("only_TSIR", False, G_SYM, "Only displace along totally symmetric coordinates.", types=B),
    # --- Output and analysis -------------------------------------------------
    OptionSpec("verbosity", "normal", G_OUT, "'quiet' (warnings only), 'normal' (progress and results) or 'debug' (all intermediate output).", choices=("quiet", "normal", "debug")),
    OptionSpec("log_file", "cma.log", G_OUT, "File receiving the full debug-level log of each run; '' to disable.", types=S),
    OptionSpec("write_molden", True, G_OUT, "Write the final normal modes to a MOLDEN file. When the level-A force constants are read from force_constants/fc_int_a.dat, the level-A displacements are rebuilt in memory for it (DispsA is not read).", types=B),
    OptionSpec("ted_check", False, G_OUT, "Print the total energy distribution of the level-B modes.", types=B),
    OptionSpec("units", "HartreeBohr", G_OUT, "Units of the written force constants: 'HartreeBohr' or 'MdyneAng'.", choices=("HartreeBohr", "MdyneAng")),
    # --- Advanced -------------------------------------------------------------
    OptionSpec("tol", 1.0e-14, G_ADV, "Matrix elements smaller than this are set to zero.", types=F, check=_non_negative),
    OptionSpec("init_bool", False, G_ADV, "Set by the program.", types=B, internal=True),
    # --- Accepted for old inputs, no current effect ----------------------
    *[
        OptionSpec(n, d, G_ADV, "Currently has no effect.", no_effect=True)
        for n, d in [
            ("autosalcs", True),
            ("benchmark_full", False),
            ("clean_house", True),
            ("dir_reap", True),
            ("disp_check", False),
            ("disp_tol", 0.0001),
            ("mode_coupling_check", False),
            ("molly_regex_b", ""),
            ("printout_rel_e", True),
            ("proj_tol", 1.0e-14),
            ("rmsd", False),
            ("spin", 1),
        ]
    ],
]
# fmt: on

OPTION_SPECS = {s.name: s for s in _SPECS}

# Removed options -> why, and what to do instead.
REMOVED_OPTIONS = {
    name: "CMA now works out which stage to run from the files on disk, so "
    "gen_disps_* and calc_* are no longer needed. Delete this option; to "
    "redo a level, move its Disps directory aside and run CMA again."
    for name in (
        "gen_disps_a",
        "gen_disps_b",
        "gen_disps_c",
        "calc_a",
        "calc_b",
        "calc_c",
    )
}

# ----------------------------------------------------------------------
# Cluster profiles
# ----------------------------------------------------------------------
# A profile is a set of option values applied before the user's own
# options. "builtin:<name>" job templates live in cluster_templates.py.
PROFILES = {
    "sapelo2_hfslab": {
        "description": "UGA GACRC Sapelo2 (Slurm) with the HFS lab's Molpro container and Psi4 conda environment; the job scripts the package used before profiles existed.",
        "options": {
            "cluster": "slurm",
            "queue": "batch",
            "job_template": "builtin:sapelo2_hfslab",
            "max_array_size": 5000,
        },
    },
    "vulcan_ccqc": {
        "description": "CCQC Vulcan (SGE); the SGE job scripts the package used before profiles existed.",
        "options": {
            "cluster": "sge",
            "job_template": "builtin:vulcan_ccqc",
        },
    },
}


def _suggest(name, candidates):
    close = difflib.get_close_matches(str(name), list(candidates), n=3, cutoff=0.6)
    return (
        " Did you mean {}?".format(" or ".join("'" + c + "'" for c in close))
        if close
        else ""
    )


class Options:
    """All settings for a Concordant Modes run.

    Pass options as keyword arguments, e.g.
    ``Options(coords="Delocalized", program_a="molpro", cluster="slurm")``.
    Run ``cma options`` or see the documentation for the full list.
    """

    def __init__(self, **kwargs):
        unknown = [k for k in kwargs if k not in OPTION_SPECS]
        if unknown:
            raise OptionsError(self._unknown_message(unknown))

        for spec in _SPECS:
            object.__setattr__(self, spec.name, _copy_default(spec.default))

        profile = kwargs.get("cluster_profile", "")
        if profile:
            if profile not in PROFILES:
                raise OptionsError(
                    "Unknown cluster_profile '{}'. Available profiles: {}.{}".format(
                        profile,
                        ", ".join(sorted(PROFILES)),
                        _suggest(profile, PROFILES),
                    )
                )
            for name, value in PROFILES[profile]["options"].items():
                object.__setattr__(self, name, _copy_default(value))

        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

        self.validate()

    # -- validation ---------------------------------------------------------
    @staticmethod
    def _unknown_message(names):
        lines = []
        for name in names:
            if name in REMOVED_OPTIONS:
                lines.append(
                    "'{}' has been removed. {}".format(name, REMOVED_OPTIONS[name])
                )
            else:
                lines.append(
                    "'{}' is not a Concordant Modes option.{}".format(
                        name, _suggest(name, OPTION_SPECS)
                    )
                )
        return "\n".join(lines) + "\nRun 'cma options' to list every option."

    def __setattr__(self, name, value):
        if name not in OPTION_SPECS:
            raise OptionsError(self._unknown_message([name]))
        object.__setattr__(self, name, value)

    def validate(self):
        """Check every option value; raise OptionsError listing all problems."""
        problems = []
        for spec in _SPECS:
            value = getattr(self, spec.name)
            msg = _check_value(spec, value)
            if msg:
                problems.append("{} = {!r}: {}".format(spec.name, value, msg))
            elif spec.choices and isinstance(value, str):
                # Normalise case so the rest of the code can compare directly.
                canonical = {c.lower(): c for c in spec.choices}[value.lower()]
                object.__setattr__(self, spec.name, canonical)

        for level in "abc":
            prog = getattr(self, "program_" + level)
            if (
                isinstance(prog, str)
                and programs.program_name(prog) not in programs.PROGRAMS
            ):
                name = programs.program_name(prog)
                problems.append(
                    "program_{} = {!r}: supported programs are {}.{}".format(
                        level,
                        prog,
                        ", ".join(programs.SUPPORTED_PROGRAMS),
                        _suggest(name, programs.PROGRAMS),
                    )
                )
            for kind in ("energy", "success"):
                name = "{}_regex_{}".format(kind, level)
                pattern = getattr(self, name)
                if pattern and isinstance(pattern, str):
                    try:
                        programs.check_regex(name, pattern)
                    except OptionsError as exc:
                        problems.append(str(exc))

        if self.cluster == "custom" and not self.custom_submit_str:
            problems.append(
                "cluster = 'custom' needs custom_submit_str (the command that submits one job)."
            )
        if self.cluster_profile and self.cluster_profile not in PROFILES:
            problems.append(
                "cluster_profile = {!r}: available profiles are {}.".format(
                    self.cluster_profile, ", ".join(sorted(PROFILES))
                )
            )

        if problems:
            raise OptionsError("Invalid options:\n  " + "\n  ".join(problems))
        return self

    # -- presets ------------------------------------------------------------
    def regexes(self, level):
        """(energy, success, gradient) patterns for level 'A', 'B' or 'C',
        falling back to the program presets for anything left empty."""
        lv = level.lower()
        prog = getattr(self, "program_" + lv)
        energy = getattr(self, "energy_regex_" + lv) or programs.energy_regex(
            prog, getattr(self, "method_" + lv)
        )
        success = getattr(self, "success_regex_" + lv) or programs.success_regex(prog)
        gradient = getattr(self, "gradient_regex_" + lv) or programs.gradient_regex(
            prog
        )
        return energy, success, gradient

    def changed(self):
        """Options that differ from their defaults (for the run log)."""
        out = {}
        for spec in _SPECS:
            value = getattr(self, spec.name)
            if value != spec.default and not spec.internal:
                out[spec.name] = value
        return out

    def no_effect_set(self):
        """Options set to a non-default value that currently do nothing."""
        return [
            s.name for s in _SPECS if s.no_effect and getattr(self, s.name) != s.default
        ]


def _copy_default(value):
    return list(value) if isinstance(value, list) else value


def _check_value(spec, value):
    if value is None:
        return None if spec.allow_none else "must not be None"
    if spec.choices:
        if not isinstance(value, str) or value.lower() not in {
            c.lower() for c in spec.choices
        }:
            return "must be one of {}.{}".format(
                ", ".join(repr(c) for c in spec.choices), _suggest(value, spec.choices)
            )
        return None
    if spec.types:
        ok = isinstance(value, spec.types)
        if bool not in spec.types and isinstance(value, bool):
            ok = False  # True/False are ints in Python; reject them for numbers
        if not ok:
            return "must be {}".format(" or ".join(t.__name__ for t in spec.types))
    if spec.check:
        return spec.check(value)
    return None


def option_reference(include_hidden=False):
    """Rows (group, name, default, help) for documentation and ``cma options``."""
    rows = []
    for spec in sorted(_SPECS, key=lambda s: GROUP_ORDER.index(s.group)):
        if spec.internal or (spec.no_effect and not include_hidden):
            continue
        rows.append((spec.group, spec.name, repr(spec.default), spec.help))
    return rows
