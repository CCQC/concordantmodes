"""Decide from the files on disk what each CMA level still needs.

For each level (B, then A) a run goes through these states::

    no Disps<L>/          -> write the displacement inputs
    some outputs missing  -> cluster="manual": stop and say what to run
                             slurm/sge: submit only those, wait, continue
                             custom: submit only those, stop
    all outputs finished  -> read energies/gradients, build force constants

Finished calculations are never rewritten or resubmitted. If ``Disps<L>``
exists but no longer matches the current zmat/template/options (for
example after editing the template), CMA stops instead of mixing old and
new calculations.
"""

import logging
import os
import re
import shutil
from pathlib import Path

from concordantmodes import fc_files, programs
from concordantmodes.cluster_templates import build_job_script
from concordantmodes.errors import CMAError, CMAWaiting
from concordantmodes.submit import Submit, numbered_dirs

logger = logging.getLogger(__name__)

LEVEL_NAMES = {"B": "low-level (B)", "A": "high-level (A)", "C": "level C"}


def fc_file(options, level):
    """Name of the force-constant file that lets CMA skip a level's
    calculations (kept in force_constants/; see fc_files)."""
    return fc_files.fc_name(level, cartesian=(level == "B" and options.cart_fc_b))


def has_fc(options, level, root):
    return fc_files.find(str(root), fc_file(options, level), quiet=True) is not None


def input_name(program):
    return programs.PROGRAMS[programs.program_name(program)]["input_file"]


def _short_list(numbers, limit=15):
    numbers = list(numbers)
    text = ", ".join(str(n) for n in numbers[:limit])
    return text + (
        " ... ({} total)".format(len(numbers)) if len(numbers) > limit else ""
    )


# ----------------------------------------------------------------------
def preflight(options, rootdir):
    """Check the run directory before any work starts and report every
    missing file at once."""
    root = Path(rootdir)
    problems = []

    if not (root / "zmat").exists():
        problems.append(
            "zmat - the geometry and coordinate definition (a ZMAT block and a cart block). "
            "See the examples directory or run 'cma init'."
        )

    if options.geom_check:  # only the zmat file is read
        if problems:
            raise CMAError(
                "Missing input file in {}:\n  - {}".format(root, problems[0])
            )
        return

    need = []
    if not has_fc(options, "B", root) and not (root / "DispsB").exists():
        need.append("B")
    if not has_fc(options, "A", root) and not (root / "DispsA").exists():
        need.append("A")
    for level in need:
        template = root / "template{}.dat".format(level)
        if not template.exists():
            problems.append(
                "{} - the {} input for program_{} = '{}', with a {{geometry}} line where "
                "the coordinates go.".format(
                    template.name,
                    LEVEL_NAMES[level],
                    level.lower(),
                    getattr(options, "program_" + level.lower()),
                )
            )

    if fc_files.find(str(root), "fc_b.dat", quiet=True) and not has_fc(
        options, "B", root
    ):
        logger.warning(
            "Found fc_b.dat, but CMA reads level-B force constants from %s. For a "
            "Cartesian Hessian (CFOUR FCMFINAL format) set cart_fc_b = True and name "
            "the file fc_cart_b.dat; for internal coordinates name it fc_int_b.dat.",
            fc_files.display(fc_file(options, "B")),
        )

    if problems:
        raise CMAError(
            "Missing input file(s) in {}:\n  - ".format(root) + "\n  - ".join(problems)
        )


# ----------------------------------------------------------------------
def prepare_level(
    options, rootdir, level, disps_name, make_tree, reap_obj, prog_name, prog
):
    """Make sure every calculation of ``level`` is finished, or stop.

    ``make_tree(dir_name)`` returns a DirectoryTree that writes this level's
    inputs into ``dir_name``; ``reap_obj`` is the level's Reap object (used
    to classify each output as done / missing / failed).
    """
    root = Path(rootdir)
    disps = root / disps_name
    os.chdir(root)  # templates and Disps directories are relative to the run directory

    if not disps.exists():
        make_tree(disps_name).run()
        dirs = numbered_dirs(disps)
        logger.info(
            "Level %s: wrote %d %s inputs to %s/",
            level,
            len(dirs),
            prog_name,
            disps_name,
        )
    else:
        _check_matches(root, disps, disps_name, make_tree, prog)
        dirs = numbered_dirs(disps)

    todo, failed = _unfinished(reap_obj, disps, dirs)
    if not todo:
        logger.info("Level %s: all %d calculations finished.", level, len(dirs))
        return

    if options.cluster == "manual":
        raise CMAWaiting(
            _manual_message(options, level, disps, dirs, todo, failed, prog_name, prog),
            level=level,
            missing=[d for d in todo if d not in failed],
            failed=failed,
        )

    if failed:
        logger.info(
            "Level %s: resubmitting %d failed calculation(s): %s",
            level,
            len(failed),
            _short_list(failed),
        )
    Submit(options, level, rootdir, prog_name, prog).run(todo)

    todo, failed = _unfinished(reap_obj, disps, dirs)
    if todo:
        raise CMAError(
            "Level {lv}: {n} calculation(s) still have no successful output after the "
            "jobs left the queue: Disps{lv} directories {dirs}. Look at their "
            "{out} and scheduler logs (Disps{lv}/slurm_logs for Slurm arrays); run "
            "CMA again to resubmit only these.".format(
                lv=level, n=len(todo), dirs=_short_list(todo), out=reap_obj.output_name
            )
        )
    logger.info("Level %s: all %d calculations finished.", level, len(dirs))


def _unfinished(reap_obj, disps, dirs):
    statuses = {d: reap_obj.check_dir(disps / str(d)) for d in dirs}
    todo = [d for d in dirs if statuses[d] != "done"]
    failed = [d for d in dirs if statuses[d] == "failed"]
    return todo, failed


def _manual_message(options, level, disps, dirs, todo, failed, prog_name, prog):
    done = len(dirs) - len(todo)
    lines = [
        "Level {}: {} of {} calculations finished.".format(level, done, len(dirs)),
    ]
    not_run = [d for d in todo if d not in failed]
    if not_run:
        lines.append(
            "  Not run yet: {} ({}).".format(len(not_run), _short_list(not_run))
        )
    if failed:
        lines.append(
            "  Failed (output present but no success/energy line): {} ({}).".format(
                len(failed), _short_list(failed)
            )
        )
    lines.append(
        "Run {inp} in each of these directories of {d}/ so that it writes output.dat "
        "next to it, then run CMA again. Finished directories are left alone.".format(
            inp=input_name(prog), d=disps.name
        )
    )
    example = disps / "sub_script.sh"
    if not example.exists():
        try:
            example.write_text(
                build_job_script(options, level, prog_name, prog, len(dirs))
            )
            lines.append(
                "An example job script for one directory was written to {}/{}.".format(
                    disps.name, example.name
                )
            )
        except CMAError:
            pass
    lines.append(
        "To have CMA submit and wait for the jobs itself, set cluster = 'slurm' or 'sge'."
    )
    return "\n".join(lines)


_NUMBER = re.compile(r"-?\d+\.\d+")

# Coordinates may differ in the last printed digits between runs: level-A
# displacements depend on the level-B force constants, which are re-read
# from fc_int_b.dat (10 decimals) on later runs.
COORD_TOL = 1.0e-6


def _same_input(old, new):
    """Same text, with decimal numbers equal to within COORD_TOL."""
    if old == new:
        return True
    old_lines, new_lines = old.splitlines(), new.splitlines()
    if len(old_lines) != len(new_lines):
        return False
    for a, b in zip(old_lines, new_lines):
        if a == b:
            continue
        # Compare token by token: a sign flip of zero (-0.0 vs 0.0) also
        # shifts the column padding.
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x == y:
                continue
            if not (_NUMBER.fullmatch(x) and _NUMBER.fullmatch(y)):
                return False
            if abs(float(x) - float(y)) > COORD_TOL:
                return False
    return True


def _check_matches(root, disps, disps_name, make_tree, prog):
    """Regenerate the inputs in a scratch directory and compare them with
    the existing Disps directory."""
    check_name = ".cma_check_" + disps_name
    check = root / check_name
    for path in (check, root / ("old" + check_name)):
        shutil.rmtree(path, ignore_errors=True)
    try:
        make_tree(check_name).run()
        expected = numbered_dirs(check)
        existing = numbered_dirs(disps)
        inp = input_name(prog)
        first_diff = None
        if expected != existing:
            first_diff = "{} has {} displacement directories but the current inputs give {}".format(
                disps_name, len(existing), len(expected)
            )
        else:
            for d in expected:
                new = (check / str(d) / inp).read_text()
                old_path = disps / str(d) / inp
                if not old_path.exists() or not _same_input(old_path.read_text(), new):
                    first_diff = (
                        "{}/{}/{} differs from what the current inputs produce".format(
                            disps_name, d, inp
                        )
                    )
                    break
    finally:
        shutil.rmtree(check, ignore_errors=True)
        shutil.rmtree(root / ("old" + check_name), ignore_errors=True)

    if first_diff:
        raise CMAError(
            "{} was made from different inputs ({}). If you changed the zmat file, a "
            "template or an option, move the old directory aside so CMA can start this "
            "level fresh, e.g.  mv {} {}_old  - then run CMA again.".format(
                disps_name, first_diff, disps_name, disps_name
            )
        )


# ----------------------------------------------------------------------
def status(options, rootdir="."):
    """Summary of each level's progress, for ``cma status``."""
    from concordantmodes.reap import Reap

    root = Path(rootdir)
    rows = []
    for level in ("B", "A"):
        lv = level.lower()
        disps = root / ("Disps" + level)
        found = fc_files.find(str(root), fc_file(options, level), quiet=True)
        if found:
            rows.append(
                (
                    level,
                    "done (force constants read from {})".format(
                        os.path.relpath(found, root)
                    ),
                )
            )
            continue
        if not disps.exists():
            rows.append((level, "not started"))
            continue
        reap = Reap(
            options,
            0,
            [],
            None,
            level,
            deriv_level=getattr(options, "deriv_level_" + lv),
        )
        dirs = numbered_dirs(disps)
        todo, failed = _unfinished(reap, disps, dirs)
        text = "{} of {} calculations finished".format(len(dirs) - len(todo), len(dirs))
        not_run = [d for d in todo if d not in failed]
        if not_run:
            text += "; not run: {}".format(_short_list(not_run))
        if failed:
            text += "; failed: {}".format(_short_list(failed))
        rows.append((level, text))
    return rows
