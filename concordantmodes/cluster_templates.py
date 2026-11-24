"""Job scripts for running displacement calculations on a cluster.

The script for a level comes from, in order of preference:

1. ``options.job_template`` pointing to a file - your own script;
2. ``options.job_template = "builtin:<profile>"`` - a group-specific
   script shipped with the package (set by ``cluster_profile``);
3. the generic Slurm or SGE script below.

Placeholders are written as ``{name}`` and only the names in
``placeholders()`` are replaced, so shell syntax such as ``${VAR}`` in your
own template is left alone.

Each script runs one calculation in its own displacement directory. For
Slurm job arrays and SGE task arrays the scheduler starts the script in the
``DispsA``/``DispsB`` directory and the script (or the array preamble added
by ``submit.py``) changes into the task's numbered directory.
"""

from pathlib import Path

from concordantmodes import programs
from concordantmodes.errors import CMAError

# Command that runs one calculation inside its displacement directory.
DEFAULT_RUN_COMMANDS = {
    "molpro": "molpro -n $NSLOTS --nouse-logfile -o output.dat input.dat",
    "psi4": "psi4 -n $NSLOTS -i input.dat -o output.dat",
    "orca": '"$(command -v orca)" input.dat > output.dat',
    "cfour": "xcfour > output.dat 2>&1",
}

GENERIC_SLURM = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={queue}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={nslots}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
{directives}
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export NSLOTS={nslots}

{job_setup}

{run_command}
"""

GENERIC_SGE = """#!/bin/bash
#$ -N {job_name}
#$ -q {queue}
#$ -S /bin/bash
#$ -cwd
#$ -l h_rt={time_limit}
{directives}
# The scheduler starts each task in the Disps directory; move into the
# task's displacement directory.
cd $SGE_O_WORKDIR/$SGE_TASK_ID
export NSLOTS=${{NSLOTS:-{nslots}}}

{job_setup}

{run_command}
"""


def run_command(options, level, prog_name):
    """The command that runs one calculation for this level."""
    custom = getattr(options, "run_command_" + level.lower())
    if custom:
        return custom
    return DEFAULT_RUN_COMMANDS[programs.program_name(prog_name)]


def placeholders(options, level, prog_name, n_jobs):
    return {
        "job_name": options.job_name,
        "queue": options.queue,
        "nslots": str(options.nslots),
        "memory": options.memory,
        "time_limit": options.time_limit,
        "directives": "\n".join(options.scheduler_directives),
        "job_setup": options.job_setup,
        "run_command": run_command(options, level, prog_name),
        "program": programs.program_name(prog_name),
        "level": level.upper(),
        "n_jobs": str(n_jobs),
    }


def fill(text, values):
    """Replace only the known {name} placeholders."""
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_job_script(options, level, prog_name, prog, n_jobs):
    """Return the job script text for one displacement calculation."""
    source = options.job_template
    values = placeholders(options, level, prog_name, n_jobs)

    if source.startswith("builtin:"):
        name = source.split(":", 1)[1]
        if name == "sapelo2_hfslab":
            from concordantmodes.sapelo_template import SapeloTemplate

            return SapeloTemplate(
                options, n_jobs, programs.program_name(prog_name), prog
            ).run()
        if name == "vulcan_ccqc":
            from concordantmodes.vulcan_template import VulcanTemplate

            if programs.program_name(prog_name) != "cfour":
                raise CMAError(
                    "The vulcan_ccqc profile only has a CFOUR job script; for {} use "
                    "cluster='sge' without the profile (generic SGE script) or your "
                    "own job_template.".format(programs.program_name(prog_name))
                )
            return VulcanTemplate(
                options, n_jobs, programs.program_name(prog_name), prog
            ).run()
        raise CMAError(
            "job_template = {!r}: unknown built-in template. Available: "
            "builtin:sapelo2_hfslab, builtin:vulcan_ccqc.".format(source)
        )

    if source:
        path = Path(source).expanduser()
        if not path.exists():
            raise CMAError(
                "job_template = {!r}: file not found (looked for {}).".format(
                    source, path.resolve()
                )
            )
        return fill(path.read_text(), values)

    if options.cluster == "slurm":
        return fill(GENERIC_SLURM, values)
    if options.cluster == "sge":
        return fill(GENERIC_SGE, values)
    # manual / custom clusters: a generic Slurm-style script is still handy
    # as a starting point for the user's own submission command.
    return fill(GENERIC_SLURM, values)
