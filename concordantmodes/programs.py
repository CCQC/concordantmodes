"""Built-in output-parsing presets for each supported program.

``Options`` uses these to fill in ``energy_regex_*``, ``success_regex_*`` and
``gradient_regex_*`` when the user leaves them empty. An explicit regex
option always wins.

Energy parsing
--------------
* Molpro: by default the summary line Molpro prints at the end of every
  job, e.g. ``CCSD(T)/cc-pVDZ energy=   -115.420205054187``, which holds
  the energy of the last method in the input. ``method_*`` selects a
  specific method's line instead (useful when the input runs several
  methods and the one you want is not last).
* Psi4: Psi4 prints different lines for every method, so CMA appends one
  line to each generated input that prints the ``CURRENT ENERGY``
  variable (the energy of the last ``energy()`` call) as
  ``CMA_ENERGY <value>``. ``method_*`` is not needed.
* ORCA: ``FINAL SINGLE POINT ENERGY``, printed for every method.
* CFOUR: ``The final electronic energy is``, printed for every method.

Sources: Molpro lines from the package's methanol example output; Psi4
lines from the Psi4 test-suite reference outputs; ORCA lines from the
ORCA manual's "Hello water" example; the CFOUR line as parsed by the
pysisyphus CFOUR interface. Gradient presets are the ones used by the
package examples.
"""

import re

from concordantmodes.errors import OptionsError

FLOAT = r"(-?\d+\.\d+)"

PSI4_ENERGY_MARKER = "CMA_ENERGY"
PSI4_ENERGY_LINE = (
    '\npsi4.print_out("\\n{marker} %.12f\\n" % psi4.variable("CURRENT ENERGY"))\n'
).format(marker=PSI4_ENERGY_MARKER)

PROGRAMS = {
    "molpro": {
        "success": r"Molpro calculation terminated",
        "energy": r"\S+/\S+\s+energy=\s+" + FLOAT,
        "methods": {
            "hf": r"!(?:RHF|UHF|HF)(?:-SCF)? STATE \d+\.\d+ Energy\s+" + FLOAT,
            "rhf": r"!RHF STATE \d+\.\d+ Energy\s+" + FLOAT,
            "uhf": r"!UHF STATE \d+\.\d+ Energy\s+" + FLOAT,
            "mp2": r"!MP2 total energy\s+" + FLOAT,
            "ccsd": r"!CCSD total energy\s+" + FLOAT,
            "ccsd(t)": r"!CCSD\(T\) total energy\s+" + FLOAT,
        },
        "gradient": [r"virial=", r"gradient"],
        "input_file": "input.dat",
    },
    "psi4": {
        "success": r"Psi4 exiting successfully",
        "energy": PSI4_ENERGY_MARKER + r"\s+" + FLOAT,
        "methods": {},
        "gradient": [r"Total Gradient", r"tstop"],
        "input_file": "input.dat",
    },
    "orca": {
        "success": r"ORCA TERMINATED NORMALLY",
        "energy": r"FINAL SINGLE POINT ENERGY\s+" + FLOAT,
        "methods": {},
        "gradient": None,
        "input_file": "input.dat",
    },
    "cfour": {
        # CFOUR prints no single "finished" banner; the final-energy line
        # only appears once the energy calculation has completed.
        "success": r"The final electronic energy is",
        "energy": r"The final electronic energy is\s+" + FLOAT,
        "methods": {},
        "gradient": None,
        "input_file": "ZMAT",
    },
}

SUPPORTED_PROGRAMS = tuple(PROGRAMS)


def program_name(program):
    """``"molpro@2021"`` -> ``"molpro"`` (the version suffix is ignored)."""
    return str(program).split("@")[0].strip().lower()


def energy_regex(program, method=""):
    prog = PROGRAMS[program_name(program)]
    if not method:
        return prog["energy"]
    key = method.strip().lower()
    if key in prog["methods"]:
        return prog["methods"][key]
    if prog["methods"]:
        raise OptionsError(
            "No built-in energy pattern for method '{}' with {}. Known methods: {}. "
            "Leave the method option empty to use {}'s final-energy line, or set "
            "energy_regex_* yourself.".format(
                method,
                program_name(program),
                ", ".join(sorted(prog["methods"])),
                program_name(program),
            )
        )
    # Programs with a single method-independent energy line ignore method.
    return prog["energy"]


def success_regex(program):
    return PROGRAMS[program_name(program)]["success"]


def gradient_regex(program):
    return PROGRAMS[program_name(program)]["gradient"]


def needs_psi4_energy_line(program, energy_pattern):
    """True when generated Psi4 inputs need the CMA_ENERGY print appended."""
    return (
        program_name(program) == "psi4" and energy_pattern == PROGRAMS["psi4"]["energy"]
    )


def check_regex(name, pattern):
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise OptionsError("{} is not a valid regular expression: {}".format(name, exc))
    return compiled
