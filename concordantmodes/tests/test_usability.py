"""Tests for option validation, program presets, input checks, job
submission planning and the automatic multi-run workflow."""

import os
import re
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from concordantmodes import programs
from concordantmodes.cma import ConcordantModes
from concordantmodes.errors import CMAError, OptionsError
from concordantmodes.options import Options, option_reference
from concordantmodes.submit import Submit, compress_ranges, range_spec
from concordantmodes import workflow
from concordantmodes.zmat import Zmat

DATA = Path(__file__).resolve().parent / "ref_data" / "usability"

# Frequencies the pre-0.2 code gives for the fake-Molpro water model below.
REFERENCE_FREQS = [1650.55, 3717.72, 3849.97]


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
def test_unknown_option_suggests_close_match():
    with pytest.raises(OptionsError, match="Did you mean 'cluster'"):
        Options(cluser="slurm")


def test_removed_option_explains_new_workflow():
    with pytest.raises(OptionsError, match="works out which stage"):
        Options(gen_disps_b=False)


def test_assigning_unknown_attribute_fails():
    options = Options()
    with pytest.raises(OptionsError, match="cart_insert_"):
        options.cart_insert = 9


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"coords": "delocalised"}, "Did you mean 'Delocalized'"),
        ({"program_a": "molpr"}, "Did you mean 'molpro'"),
        ({"nslots": "4"}, "nslots = '4': must be int"),
        ({"disp_b": True}, "disp_b = True"),
        ({"cluster": "custom"}, "needs custom_submit_str"),
        ({"energy_regex_a": "(T total"}, "not a valid regular expression"),
        ({"deriv_level_b": 2}, "must be 0"),
    ],
)
def test_invalid_values(kwargs, message):
    with pytest.raises(OptionsError, match=re.escape(message)):
        Options(**kwargs)


def test_choices_are_case_insensitive_and_normalised():
    assert Options(coords="delocalized", cluster="SLURM").coords == "Delocalized"
    assert Options(cluster="SLURM").cluster == "slurm"


def test_profile_applies_and_user_options_win():
    options = Options(cluster_profile="sapelo2_hfslab", max_array_size=100)
    assert options.cluster == "slurm"
    assert options.job_template == "builtin:sapelo2_hfslab"
    assert options.max_array_size == 100


def test_every_option_is_documented():
    rows = option_reference()
    assert len(rows) > 60
    assert all(help_text for _, _, _, help_text in rows)


# ----------------------------------------------------------------------
# Program presets
# ----------------------------------------------------------------------
def test_molpro_presets_on_real_output():
    text = (DATA / "molpro_ccsdt_tail.out").read_text()
    options = Options(program_a="molpro")
    energy, success, _ = options.regexes("A")
    assert re.search(success, text)
    assert float(re.findall(energy, text)[0]) == pytest.approx(-115.420205054187)

    for method, value in [("rhf", -115.049111158487), ("ccsd(t)", -115.420205054187)]:
        pattern = Options(program_a="molpro", method_a=method).regexes("A")[0]
        assert float(re.findall(pattern, text)[0]) == pytest.approx(value)


def test_unknown_method_is_reported():
    with pytest.raises(OptionsError, match="Known methods"):
        Options(program_a="molpro", method_a="casscf").regexes("A")


def test_explicit_regex_overrides_preset():
    options = Options(program_b="psi4", energy_regex_b=r"Giraffe\s+(-?\d+\.\d+)")
    assert options.regexes("B")[0] == r"Giraffe\s+(-?\d+\.\d+)"
    assert not programs.needs_psi4_energy_line("psi4", options.regexes("B")[0])


@pytest.mark.parametrize(
    "program, text, value",
    [
        (
            "orca",
            "FINAL SINGLE POINT ENERGY       -75.959340303174\n****ORCA TERMINATED NORMALLY****",
            -75.959340303174,
        ),
        (
            "psi4",
            "CMA_ENERGY -76.241123456789\n*** Psi4 exiting successfully. Buy a developer a beer!",
            -76.241123456789,
        ),
        (
            "cfour",
            "  The final electronic energy is       -76.24112345678 a.u.",
            -76.24112345678,
        ),
    ],
)
def test_other_program_presets(program, text, value):
    energy, success, _ = Options(program_a=program).regexes("A")
    assert re.search(success, text)
    assert float(re.findall(energy, text)[0]) == pytest.approx(value)


# ----------------------------------------------------------------------
# zmat checks
# ----------------------------------------------------------------------
def _parse(tmp_path, text, coords="Custom"):
    (tmp_path / "zmat").write_text(text)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        Zmat(Options(coords=coords)).run()
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: "", "is empty"),
        (lambda t: t.replace("cart end", ""), "never closed"),
        (lambda t: t.replace("2 3\n", "2 7\n"), "refers to atom 7"),
        (lambda t: t.replace("2 3\n", "2 x3\n"), "not a recognised coordinate line"),
        (
            lambda t: t.replace("-0.12550454", "-0.12a50454"),
            "not a valid Cartesian line",
        ),
    ],
)
def test_zmat_errors_name_the_line(tmp_path, edit, message):
    good = (DATA / "water_zmat").read_text()
    with pytest.raises(CMAError, match=message):
        _parse(tmp_path, edit(good))


def test_zmat_without_trailing_newline(tmp_path):
    _parse(tmp_path, (DATA / "water_zmat").read_text().rstrip("\n"))


# ----------------------------------------------------------------------
# Job submission planning
# ----------------------------------------------------------------------
def test_array_planning():
    assert Submit.plan_array_chunks(10000, 5000) == [(0, 5000), (5000, 5000)]
    assert Submit.plan_array_chunks(5001, 5000) == [(0, 5000), (5000, 1)]
    blocks = Submit.array_blocks([3, 6, 7, 5002], 5000)
    assert blocks == [(0, [2, 5, 6]), (5000, [1])]
    assert all(max(ids) < 5000 for _, ids in blocks)
    assert compress_ranges([1, 2, 3, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]
    assert range_spec([0, 1, 2, 5]) == "0-2,5"


# ----------------------------------------------------------------------
# Automatic workflow, end to end, with a fake Molpro
# ----------------------------------------------------------------------
@pytest.fixture
def water_run(tmp_path, monkeypatch):
    shutil.copy(DATA / "water_zmat", tmp_path / "zmat")
    shutil.copy(DATA / "templateA.dat", tmp_path / "templateA.dat")
    shutil.copy(DATA / "templateB.dat", tmp_path / "templateB.dat")
    fake = tmp_path / "bin" / "molpro"
    fake.parent.mkdir()
    fake.write_text(
        "#!{}\n".format(sys.executable)
        + (DATA / "fake_molpro.py").read_text().split("\n", 1)[1]
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.chdir(tmp_path)
    return tmp_path, fake


def _run_all(fake, disps):
    import subprocess

    for d in sorted(Path(disps).iterdir()):
        if d.is_dir() and not (d / "output.dat").exists():
            subprocess.run(
                [str(fake), "-o", "output.dat", "input.dat"], cwd=d, check=True
            )


def _cma(**kw):
    cma = ConcordantModes(
        Options(coords="Delocalized", log_file="", verbosity="quiet", **kw)
    )
    return cma, cma.run()


def test_manual_workflow_matches_reference(water_run):
    root, fake = water_run
    _, status = _cma()
    assert status == "waiting" and (root / "DispsB").is_dir()
    assert workflow.status(Options(), root)[0][1].startswith("0 of 13")

    # Rerunning before anything ran changes nothing.
    _, status = _cma()
    assert status == "waiting"

    _run_all(fake, root / "DispsB")
    (root / "DispsB" / "4" / "output.dat").write_text("crashed\n")  # one failed job
    _, status = _cma()
    assert status == "waiting" and not (root / "DispsA").exists()
    assert "failed: 4" in workflow.status(Options(), root)[0][1]

    (root / "DispsB" / "4" / "output.dat").unlink()
    _run_all(fake, root / "DispsB")
    _, status = _cma()
    assert status == "waiting" and (root / "DispsA").is_dir()

    _run_all(fake, root / "DispsA")
    cma, status = _cma()
    assert status == "finished"
    assert (root / "force_constants" / "fc_cart_a.dat").exists()

    # A finished run can be repeated.
    cma, status = _cma()
    assert status == "finished"


def test_final_frequencies(water_run, caplog):
    root, fake = water_run
    import logging

    for _ in range(2):
        _cma()
        for level in ("DispsB", "DispsA"):
            if (root / level).exists():
                _run_all(fake, root / level)
    with caplog.at_level(logging.INFO, logger="concordantmodes"):
        cma = ConcordantModes(Options(coords="Delocalized", log_file=""))
        assert cma.run() == "finished"
    text = caplog.text
    final = text[text.index("Final (CMA) harmonic frequencies") :]
    freqs = [
        float(x) for x in re.findall(r"frequency #\s*\d+:\s+(-?\d+\.\d+)", final)[:3]
    ]
    assert freqs == pytest.approx(REFERENCE_FREQS, abs=0.01)


def test_changed_template_is_detected(water_run):
    root, fake = water_run
    _cma()
    (root / "templateB.dat").write_text(
        (root / "templateB.dat").read_text().replace("cc-pvdz", "aug-cc-pvdz")
    )
    with pytest.raises(CMAError, match="made from different inputs"):
        _cma()


def test_missing_inputs_listed_together(water_run):
    root, _ = water_run
    (root / "templateA.dat").unlink()
    (root / "templateB.dat").unlink()
    with pytest.raises(CMAError) as err:
        _cma()
    assert "templateA.dat" in str(err.value) and "templateB.dat" in str(err.value)


def test_template_needs_geometry_position(water_run):
    root, _ = water_run
    text = (root / "templateB.dat").read_text().replace("{geometry}\n", "")
    (root / "templateB.dat").write_text(text)
    with pytest.raises(CMAError, match="does not know where to put the geometry"):
        _cma()


def _finish(root, fake):
    for _ in range(3):
        _cma()
        for level in ("DispsB", "DispsA"):
            if (root / level).exists():
                _run_all(fake, root / level)
    _, status = _cma()
    assert status == "finished"


def test_fc_int_a_is_used_and_molden_rebuilt(water_run):
    root, fake = water_run
    _finish(root, fake)
    molden = (root / "MOLDEN").read_text()
    cart_a = (root / "force_constants" / "fc_cart_a.dat").read_text()

    # fc_int_a.dat wins: a broken DispsA output is ignored on rerun.
    (root / "DispsA" / "3" / "output.dat").write_text("crashed\n")
    (root / "MOLDEN").unlink()
    _, status = _cma()
    assert status == "finished"
    assert (
        root / "MOLDEN"
    ).read_text() == molden  # rebuilt from in-memory displacements
    assert (root / "force_constants" / "fc_cart_a.dat").read_text() == cart_a

    # Works with no DispsA at all.
    shutil.rmtree(root / "DispsA")
    (root / "MOLDEN").unlink()
    _, status = _cma()
    assert status == "finished" and (root / "MOLDEN").read_text() == molden
    assert not (root / "DispsA").exists()  # nothing is written to disk
    assert workflow.status(Options(), root)[1][1].startswith(
        "done (force constants read"
    )


def test_write_molden_false(water_run):
    root, fake = water_run
    _finish(root, fake)
    (root / "MOLDEN").unlink()
    _, status = _cma(write_molden=False)
    assert status == "finished" and not (root / "MOLDEN").exists()


def test_gradient_saved_for_second_order(water_run):
    root, fake = water_run
    fcdir = root / "force_constants"
    for _ in range(3):
        _cma(second_order=True)
        for level in ("DispsB", "DispsA"):
            if (root / level).exists():
                _run_all(fake, root / level)
    _, status = _cma(second_order=True)
    assert status == "finished"
    assert (fcdir / "fc_int_a.grad").exists() and (fcdir / "fc_cart_a.grad").exists()
    ref = (fcdir / "fc_cart_a.dat").read_text()

    # Rerun reads fc_int_a.dat + fc_int_a.grad: same Cartesian result.
    shutil.rmtree(root / "DispsA")
    _, status = _cma(second_order=True)
    assert status == "finished"
    assert (fcdir / "fc_cart_a.dat").read_text() == ref


def test_old_layout_files_are_still_read(water_run):
    root, fake = water_run
    _finish(root, fake)
    ref = (root / "force_constants" / "fc_cart_a.dat").read_text()
    # Move everything back to the run directory, as older versions wrote it.
    for f in (root / "force_constants").iterdir():
        f.rename(root / f.name)
    (root / "force_constants").rmdir()
    shutil.rmtree(root / "DispsA")
    shutil.rmtree(root / "DispsB")
    _, status = _cma()
    assert status == "finished"
    assert (root / "force_constants" / "fc_cart_a.dat").read_text() == ref
