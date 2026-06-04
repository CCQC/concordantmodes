from pathlib import Path
import shutil
import copy


class DirectoryTree:
    PROG_LIST = {"molpro", "psi4", "cfour", "orca"}

    INSERTION_MAP = {
        "A": "cart_insert_a",
        "B": "cart_insert_b",
        "C": "cart_insert_c",
    }

    def __init__(
        self,
        prog_name,
        zmat,
        ref_geom,
        cma_level,
        p_disp,
        m_disp,
        options,
        indices,
        symm_obj,
        template,
        dir_name,
        deriv_level=0,
    ):
        self.prog_name = prog_name
        self.zmat = zmat
        self.ref_geom = ref_geom
        self.cma_level = cma_level
        self.p_disp = p_disp
        self.m_disp = m_disp
        self.options = options
        self.indices = indices
        self.symm_obj = symm_obj
        self.template = Path(template)
        self.dir_name = Path(dir_name)
        self.deriv_level = deriv_level

        try:
            self.insertion_index = getattr(self.options, self.INSERTION_MAP[cma_level])
        except KeyError:
            raise RuntimeError("cma_level must be A, B, or C")

    def run(self):
        root = Path.cwd()

        if self.prog_name not in self.PROG_LIST:
            raise RuntimeError(f"Unsupported program: {self.prog_name}")

        data = self.template.read_text().splitlines(keepends=True)
        n_atoms = len(self.zmat.atom_list)

        # Detect optional files
        self.init = (root / "initden.dat").exists()
        self.genbas = (root / "GENBAS").exists()
        self.ecp = (root / "ECPDATA").exists()
        self.sub = (root / "sub_script.sh").exists()

        # Backup existing directory
        old_dir = root / f"old{self.dir_name}"
        if old_dir.exists():
            shutil.rmtree(old_dir)

        if self.dir_name.exists():
            shutil.copytree(self.dir_name, old_dir)
            shutil.rmtree(self.dir_name)

        self.dir_name.mkdir()
        Path.chdir = lambda p: None  # avoid lint complaints

        inp = "ZMAT" if self.prog_name == "cfour" else "input.dat"

        # Generate inputs
        if self.deriv_level == 0:
            self._run_energy(data, n_atoms, inp)
        elif self.deriv_level == 1:
            self._run_gradient(data, n_atoms, inp)
        else:
            raise RuntimeError("Only energy and gradient derivatives supported")

    def _run_energy(self, data, n_atoms, inp):
        self.make_input(data, self.ref_geom, n_atoms, inp, "1")

        indices = self._resolve_indices()

        direc = 2
        for i, j in indices:
            self.make_input(data, self.p_disp[i, j], n_atoms, inp, str(direc))
            self.make_input(data, self.m_disp[i, j], n_atoms, inp, str(direc + 1))
            direc += 2

    def _run_gradient(self, data, n_atoms, inp):

        self.make_input(data, self.ref_geom, n_atoms, inp, "1")

        direc = 2

        for idx in self.indices:
            self.make_input(data, self.p_disp[idx[0]], n_atoms, inp, str(direc))
            self.make_input(data, self.m_disp[idx[0]], n_atoms, inp, str(direc + 1))
            direc += 2

    def _resolve_indices(self):
        if self.symm_obj.symtext is not None and self.options.exploit_pm_symm:
            if self.options.only_TSIR:
                return self.symm_obj.indices_by_irrep[0]
            return self.symm_obj.indices_by_irrep.reshape((-1, 2))

        return self.indices

    def make_input(self, data, geom, n_atoms, inp, direc):
        path = self.dir_name / str(direc)
        path.mkdir(parents=True, exist_ok=True)

        space = "" if self.prog_name == "cfour" else " "

        new_data = data.copy()

        if self.insertion_index == -1:
            raise RuntimeError("Invalid insertion index")

        for i in range(n_atoms):
            atom = self.zmat.atom_list[i]
            x, y, z = geom[i]
            line = f"{space}{atom}{x:16.10f}{y:16.10f}{z:16.10f}\n"
            new_data.insert(self.insertion_index + i, line)

        (path / inp).write_text("".join(new_data))
        
        self.new_data = new_data
        
        self._copy_support_files(path)

    def _copy_support_files(self, path):
        root = Path.cwd().parent

        files = {
            "initden.dat": self.init,
            "GENBAS": self.genbas,
            "ECPDATA": self.ecp,
            "sub_script.sh": self.sub,
        }

        for fname, exists in files.items():
            if exists:
                shutil.copy(root / fname, path)
