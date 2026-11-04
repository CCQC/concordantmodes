# concordantmodes

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Concordant Modes (CMA) computes harmonic vibrational frequencies at a high level of theory with linear scaling. Normal modes from a cheap level of theory (level B) are used to choose the displacements, so the expensive method (level A) is needed for far fewer calculations. The method is described in *J. Am. Chem. Soc.* 2022, **144**, 23271–23274; *J. Chem. Theory Comput.* 2024, **20**, 10886–10898; and *J. Phys. Chem. A* 2026, **130**, 3249–3260.

## Installation

Python 3.10 or newer.

```bash
conda env create -f devtools/conda-envs/base.yaml   # optional: a ready environment
conda activate concordantmodes
pip install -e .                 # from this directory; installs the `cma` command
pip install -e ".[symmetry]"     # optional: MolSym, needed only for molsym_symmetry = True
```

## Quickstart

```bash
mkdir water && cd water
cma init --program molpro    # writes zmat, templateA.dat, templateB.dat, cma.toml
# edit zmat (your molecule) and the templates (method, basis), then:
cma run
```

`cma run` does whatever is needed next and can be run as often as you like:

1. It writes the level-B displacement inputs to `DispsB/1`, `DispsB/2`, ...
2. With the default `cluster = "manual"` it stops and tells you which inputs to run. Run each `input.dat` so that it writes `output.dat` next to it, then `cma run` again.
3. It reads the level-B results, writes the level-A inputs to `DispsA`, and stops again.
4. Once those are done, `cma run` prints the final frequencies.

Finished calculations are never redone; only missing or failed ones are run again. `cma status` shows the progress of each level, and `cma options` lists every option.

With `cluster = "slurm"` or `"sge"` CMA submits the calculations itself (as job arrays), waits for them, and finishes in a single `cma run`. On Sapelo2 with the HFS lab setup, `cluster_profile = "sapelo2_hfslab"` sets everything up.

A `main.py` works as well:

```python
from concordantmodes import ConcordantModes, Options

options = Options(coords="Delocalized", program_a="molpro", program_b="molpro")
ConcordantModes(options).run()   # returns "finished" or "waiting"
```

## Input files

| File | Contents |
|---|---|
| `zmat` | The Cartesian geometry (`cart begin` ... `cart end`) and the internal coordinates (`ZMAT begin` ... `ZMAT end`). A second geometry after a `---` line is used for level A. |
| `templateB.dat`, `templateA.dat` | Inputs for `program_b` / `program_a` (molpro, psi4, orca or cfour). A `{geometry}` line is replaced by each displaced geometry (in Bohr), `{natoms}` by the number of atoms. |
| `cma.toml` | Options, one `name = value` per line. |

CMA knows how each program reports a finished calculation and its energy, so no regular expressions are needed for Molpro, Psi4, ORCA or CFOUR. `energy_regex_*` and `success_regex_*` still override the presets.

To skip level B (CMA-0B), put its Hessian in `force_constants/` as `fc_int_b.dat` (internal coordinates) or, with `cart_fc_b = True`, `fc_cart_b.dat` (Cartesian, CFOUR FCMFINAL format).

## Output

The console shows the level-B and final frequencies, the total energy distribution and the frequency shifts (`verbosity = "quiet"` or `"debug"` changes how much). The complete record is always in `cma.log`. The normal modes go to `MOLDEN` (`write_molden = false` turns it off). All force constants and gradients go in the `force_constants/` folder: the final Cartesian force constants in `fc_cart_a.dat`, and the level-B and level-A results in `fc_int_b.dat`/`.grad` and `fc_int_a.dat`/`.grad`. Later runs read these and ignore the `Disps` folders, so delete `force_constants/fc_int_a.dat` (and `.grad`) to pick up rerun level-A calculations. Files from older versions in the run directory are still found.

## Documentation and examples

- `docs/source` — quickstart, zmat format, templates and output parsing, running on clusters, and the full options reference (build with `sphinx-build docs/source docs/build`).
- `examples/0_quickstart` — the files `cma init` writes, for Molpro and Psi4.
- `examples/` — water, methanol, ammonia, cyclopropane and benzene with different coordinate systems, programs and CMA variants.

## Development

```bash
pip install -e ".[tests]"
pytest
```

### Upgrading from older versions

- `gen_disps_a/b` and `calc_a/b` are gone: CMA works out the stage from the files on disk. Remove them from `main.py`; CMA says so if they are still there.
- Unknown option names are now errors instead of being silently ignored.
- `cluster` now defaults to `"manual"` (it was `"sge"`).
- The Sapelo2 and Vulcan job scripts are now the `sapelo2_hfslab` and `vulcan_ccqc` cluster profiles; without a profile, generic Slurm/SGE scripts built from `nslots`, `memory`, `time_limit`, `job_setup` etc. are used.
