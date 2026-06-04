import numpy as np
import os
import re
from pathlib import Path
from numpy.linalg import pinv

from concordantmodes.s_vectors import SVectors


class Reap:
    REGEX_MAP = {
        "A": ("gradient_regex_a", "energy_regex_a", "success_regex_a"),
        "B": ("gradient_regex_b", "energy_regex_b", "success_regex_b"),
        "C": ("gradient_regex_c", "energy_regex_c", "success_regex_c"),
    }

    def __init__(
        self,
        options,
        num_deg_free,
        indices,
        symm_obj,
        cma_level,
        deriv_level=0,
        disp=None,
        proj=None,
        zmat=None,
    ):
        self.options = options
        self.num_deg_free = num_deg_free
        self.indices = indices
        self.symm_obj = symm_obj
        self.deriv_level = deriv_level
        self.cma_level = cma_level
        self.disp = disp
        self.proj = proj if proj is not None else np.array([])
        self.zmat = zmat

        self.fail_list = []
        self.ref_grad = []

        self._init_regex()

    def _init_regex(self):
        try:
            grad_key, energy_key, success_key = self.REGEX_MAP[self.cma_level]
        except KeyError:
            raise RuntimeError("cma_level must be A, B, or C")

        self.energy_regex = getattr(self.options, energy_key)
        self.success_regex = getattr(self.options, success_key)

        if self.deriv_level:
            self.gradient_regex = getattr(self.options, grad_key)

        # Compile once
        self.energy_re = re.compile(self.energy_regex)
        self.success_re = re.compile(self.success_regex)

        if self.deriv_level:
            self.grad_re1 = re.compile(self.gradient_regex[0])
            self.grad_re2 = re.compile(self.gradient_regex[1])

    # -------------------------
    # Core runner
    # -------------------------
    def run(self):
        if self.deriv_level:
            return self._run_gradients()
        return self._run_energies()

    # -------------------------
    # Energy workflow
    # -------------------------
    def _run_energies(self):
        print(os.getcwd())
        print("Checking energies...")

        ref_en = self._read_energy(1)
        print(f"Reference energy: {ref_en:.10f}")

        size = self.num_deg_free
        p_en = np.zeros((size, size))
        m_en = np.zeros((size, size))

        indices = self._resolve_indices()

        direc = 2
        results = []

        for i, j in indices:
            e_plus = self._read_energy(direc)
            p_en[i, j] = e_plus
            results.append(((i, j), "plus", e_plus, direc))

            e_minus = self._read_energy(direc + 1)
            m_en[i, j] = e_minus
            results.append(((i, j), "minus", e_minus, direc + 1))

            print(f"[{i},{j}] Δ+ = {e_plus - ref_en:.9f}")
            print(f"[{i},{j}] Δ- = {e_minus - ref_en:.9f}")

            direc += 2

        self.p_en_array = p_en
        self.m_en_array = m_en
        self.ref_en = ref_en

        self._check_failures()

    # -------------------------
    # Gradient workflow
    # -------------------------
    def _run_gradients(self):
        p_list = []
        m_list = []

        ref_grad = self._read_gradient(1)

        for idx in self.indices:
            i = idx[0]

            p_grad = self._read_gradient(2 * i + 2)
            m_grad = self._read_gradient(2 * i + 3)

            p_list.append(p_grad)
            m_list.append(m_grad)

        self.ref_grad = np.array(ref_grad)
        self.p_grad_array = np.array(p_list)
        self.m_grad_array = np.array(m_list)

        if not self.options.cart_fc_b:
            self._maybe_convert_gradients()

        self._check_failures()

    # -------------------------
    # Helpers
    # -------------------------
    def _resolve_indices(self):
        if self.symm_obj.symtext and self.options.exploit_pm_symm:
            if self.options.only_TSIR:
                return self.symm_obj.indices_by_irrep[0]
            return self.symm_obj.indices_by_irrep.reshape((-1, 2))
        return self.indices

    def _read_energy(self, direc):

        direc = Path(str(direc))

        data = self._read_file(direc, "output.dat")

        if not self.success_re.search(data):
            self.fail_list.append(str(direc))
            return 0.0

        return float(self.energy_re.findall(data)[0])

    def _read_gradient(self, direc):
        fname = "output.xml" if self.options.program_b == "molpro" else "output.dat"
        lines = self._read_file(direc, fname, lines=True)

        start = next(i for i, l in enumerate(lines) if self.grad_re1.search(l)) + 1
        end = next(
            i for i, l in enumerate(lines[start:], start) if self.grad_re2.search(l)
        )

        grad = [
            line.split()[-3:]
            for line in lines[start:end]
            if re.search(r"(\s*-?\d+\.\d+){3}", line)
        ]

        return np.array(grad, dtype=float).flatten()

    def _read_file(self, direc, filename, lines=False):
        path = Path(str(direc)) / filename
        print(path)
        with open(path, "r") as f:
            return f.readlines() if lines else f.read()

    def _maybe_convert_gradients(self):
        if not (
            self.options.conv_grad
            and self.zmat is not None
            and len(self.proj)
            and self.disp is not None
            and not self.deriv_level
        ):
            return

        svec = SVectors(self.zmat, self.options)

        p_grad_buff = []
        m_grad_buff = []

        for i in range(len(self.indices)):
            svec.run(self.disp.p_disp[i], False)
            A = pinv(svec.B) @ self.proj
            p_grad_buff.append((self.p_grad_array[i].T @ A).T)

            svec.run(self.disp.m_disp[i], False)
            A = pinv(svec.B) @ self.proj
            m_grad_buff.append((self.m_grad_array[i].T @ A).T)

        self.p_grad_array = np.array(p_grad_buff)
        self.m_grad_array = np.array(m_grad_buff)

    def _check_failures(self):
        if self.fail_list:
            raise RuntimeError(f"Failed jobs: {self.fail_list}")
