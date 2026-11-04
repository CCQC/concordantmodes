"""The ``cma`` command.

    cma init      write a starter cma.toml and input templates here
    cma run       run or continue the calculation in this directory
    cma status    show what each level still needs
    cma options   list every option with its default and meaning

Options are read from ``cma.toml`` (one ``name = value`` line per option);
``--set name=value`` overrides them for a single command. A ``main.py``
that builds ``Options(...)`` and calls ``ConcordantModes(...).run()`` keeps
working too.

Exit codes for ``cma run``: 0 finished, 3 waiting for calculations, 1 error.
"""

import argparse
import ast
import os
import sys
import textwrap
from pathlib import Path

from concordantmodes.errors import CMAError, OptionsError

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

EXIT_FINISHED, EXIT_ERROR, EXIT_WAITING = 0, 1, 3

STARTER_TOML = """\
# Concordant Modes options. Run 'cma options' for the full list.
# Values use TOML syntax: strings in quotes, true/false, numbers, [lists].

coords = "Delocalized"        # ZMAT, Custom or Delocalized

# Low-level (B) and high-level (A) programs: molpro, psi4, orca or cfour.
program_b = "{program}"
program_a = "{program}"
# method_a = "ccsd(t)"        # Molpro only, if the input runs several methods

# How the displacement calculations get run:
#   "manual" - CMA writes the inputs; you run them and then 'cma run' again
#   "slurm" / "sge" - CMA submits, waits and continues by itself
cluster = "manual"
# cluster_profile = "sapelo2_hfslab"   # your group's cluster settings
# nslots = 4
# memory = "16GB"
# time_limit = "12:00:00"
# job_setup = "module load Molpro"
"""

TEMPLATES = {
    "molpro": """\
***,CMA displacement
memory,500,m
geomtyp=xyz
bohr
geometry={{
{{natoms}}
CMA displacement
{{geometry}}
}}

basis={basis}
{{rhf}}
{{{method}}}
""",
    "psi4": """\
memory 4 gb

molecule {{
0 1
units bohr
no_reorient
no_com
{{geometry}}
}}

set basis {basis}

energy('{method}')
""",
    "orca": """\
! {method} {basis} TightSCF Bohrs
* xyz 0 1
{{geometry}}
*
""",
    "cfour": """\
CMA displacement
{{geometry}}

*CFOUR(CALC={method},BASIS={basis},COORD=CARTESIAN,UNITS=BOHR)

""",
}

ZMAT_SKELETON = """\
ZMAT begin
1 2
2 3
ZMAT end

cart begin
     H        -0.00000000    -1.42462540     0.99592408
     O         0.00000000     0.00000000    -0.12550454
     H        -0.00000000     1.42462540     0.99592408
cart end
"""
# Water is only an example: replace the connectivity and geometry.


def _parse_set(items):
    values = {}
    for item in items or []:
        if "=" not in item:
            raise OptionsError("--set expects name=value, got {!r}".format(item))
        name, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # plain string
        values[name.strip()] = value
    return values


def load_options(path, overrides=None):
    from concordantmodes.options import Options

    values = {}
    path = Path(path)
    if path.exists():
        with open(path, "rb") as fh:
            try:
                values = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise OptionsError("{} is not valid TOML: {}".format(path, exc))
    else:
        raise OptionsError(
            "{} not found in {}. Run 'cma init' to create one, or point --input at "
            "your options file.".format(path, Path.cwd())
        )
    values.update(overrides or {})
    return Options(**values)


# ----------------------------------------------------------------------
def cmd_init(args):
    from concordantmodes import programs

    program = programs.program_name(args.program)
    if program not in TEMPLATES:
        raise OptionsError("--program must be one of: " + ", ".join(TEMPLATES))
    defaults = {
        "molpro": ("cc-pvdz", "cc-pvtz", "ccsd(t)"),
        "psi4": ("cc-pvdz", "cc-pvtz", "ccsd(t)"),
        "orca": ("cc-pvdz", "cc-pvtz", "CCSD(T)"),
        "cfour": ("PVDZ", "PVTZ", "CCSD(T)"),
    }[program]
    files = {
        "cma.toml": STARTER_TOML.format(program=program),
        "templateB.dat": TEMPLATES[program].format(
            basis=defaults[0], method=defaults[2]
        ),
        "templateA.dat": TEMPLATES[program].format(
            basis=defaults[1], method=defaults[2]
        ),
        "zmat": ZMAT_SKELETON,
    }
    for name, text in files.items():
        path = Path(name)
        if path.exists():
            print("kept existing {}".format(name))
        else:
            path.write_text(text)
            print("wrote {}".format(name))
    print(textwrap.dedent("""
            Next: edit zmat (your molecule) and the two templates (method, basis,
            memory), check cma.toml, then run 'cma run'. The {geometry} line in each
            template is replaced by the coordinates of every displacement."""))
    return EXIT_FINISHED


def cmd_run(args):
    from concordantmodes.cma import ConcordantModes

    options = load_options(args.input, _parse_set(args.set))
    status = ConcordantModes(options).run()
    return EXIT_FINISHED if status == "finished" else EXIT_WAITING


def cmd_status(args):
    from concordantmodes import workflow

    options = load_options(args.input, _parse_set(args.set))
    for level, text in workflow.status(options, "."):
        print("Level {}: {}".format(level, text))
    final = Path("force_constants") / "fc_cart_a.dat"
    if final.exists():
        print("Final force constants: {} (CMA finished).".format(final))
    return EXIT_FINISHED


def cmd_options(args):
    from concordantmodes.options import option_reference

    rows = option_reference(include_hidden=args.all)
    if args.name:
        rows = [r for r in rows if args.name.lower() in r[1].lower()]
        if not rows:
            print("No option matching {!r}.".format(args.name))
            return EXIT_ERROR
    width = 78
    group = None
    for g, name, default, help_text in rows:
        if g != group:
            print("\n== {} ==".format(g))
            group = g
        print("{} (default {})".format(name, default))
        print(
            textwrap.fill(
                help_text, width, initial_indent="    ", subsequent_indent="    "
            )
        )
    return EXIT_FINISHED


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cma",
        description="Concordant Modes: accurate harmonic frequencies from cheap normal modes.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="show Python tracebacks on errors"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter cma.toml, templates and zmat")
    p.add_argument("--program", default="molpro", help="molpro, psi4, orca or cfour")
    p.set_defaults(func=cmd_init)

    for name, func, text in (
        ("run", cmd_run, "run or continue the calculation in this directory"),
        ("status", cmd_status, "show what each level still needs"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "-i", "--input", default="cma.toml", help="options file (default cma.toml)"
        )
        p.add_argument(
            "--set",
            action="append",
            metavar="NAME=VALUE",
            help="override one option, e.g. --set verbosity=debug (repeatable)",
        )
        p.set_defaults(func=func)

    p = sub.add_parser("options", help="list options with defaults and descriptions")
    p.add_argument("name", nargs="?", help="only options whose name contains this")
    p.add_argument(
        "--all", action="store_true", help="include options that currently do nothing"
    )
    p.set_defaults(func=cmd_options)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CMAError as exc:
        if args.debug:
            raise
        if not getattr(exc, "logged", False):  # ConcordantModes.run already showed it
            print("Error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
