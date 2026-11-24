"""Submit displacement calculations to a cluster and wait for them.

``Submit.run(dirs)`` submits the numbered directories ``dirs`` of
``Disps<level>`` (all of them when ``dirs`` is None), so the workflow can
resubmit only calculations that are missing or failed.

* ``cluster="slurm"``: job arrays (``slurm_array=True``, split into arrays of
  at most ``max_array_size`` tasks) or one ``sbatch`` per directory. CMA
  waits, polling ``sacct`` every ``poll_interval`` seconds.
* ``cluster="sge"``: one ``qsub -t`` task array per contiguous run of
  directories; CMA waits, polling ``qacct``.
* ``cluster="custom"``: runs ``custom_submit_str <dir>/sub_script.sh`` in
  each directory, then stops (``CMAWaiting``) because completion cannot be
  tracked; run CMA again when the jobs are done.
"""

import logging
import re
import subprocess
import time
from pathlib import Path

from concordantmodes.cluster_templates import build_job_script
from concordantmodes.errors import CMAError, CMAWaiting

logger = logging.getLogger(__name__)

# Slurm states in which a job (or array task) has not finished yet.
ACTIVE_SLURM_STATES = (
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
)

# Inserted after the #SBATCH header in array mode. Each array task points
# SLURM_SUBMIT_DIR at its own displacement directory and changes into it,
# so any template line that uses SLURM_SUBMIT_DIR (cd, output paths, bind
# mounts) works unchanged. CMA_OFFSET is added to the task index.
SLURM_ARRAY_PREAMBLE = """# --- Concordant Modes job-array mapping ---
CMA_DISP=$(( ${CMA_OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))
export SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR/$CMA_DISP"
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Array task $SLURM_ARRAY_TASK_ID -> displacement $CMA_DISP"
"""

SBATCH_ID = re.compile(r"Submitted\s*batch\s*job(?:-array)?\s*(\d+)")
QSUB_ID = re.compile(r"Your\s*job(?:\-array)?\s*(\d+)")


def numbered_dirs(path):
    """Sorted numbers of the numbered subdirectories of ``path``."""
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(
        int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()
    )


def compress_ranges(numbers):
    """[1,2,3,7,9,10] -> [(1,3),(7,7),(9,10)]"""
    ranges = []
    for n in sorted(numbers):
        if ranges and n == ranges[-1][1] + 1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])
    return [tuple(r) for r in ranges]


def range_spec(numbers):
    """[0,1,2,5] -> '0-2,5' (Slurm --array syntax)."""
    return ",".join(
        str(a) if a == b else "{}-{}".format(a, b) for a, b in compress_ranges(numbers)
    )


def insert_after_header(script, block):
    """Insert ``block`` after the shebang and leading #SBATCH/#$ lines."""
    lines = script.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (
        lines[i].startswith("#!")
        or lines[i].startswith("#SBATCH")
        or lines[i].startswith("#$")
        or lines[i].startswith("##SBATCH")
        or not lines[i].strip()
    ):
        i += 1
    return "".join(lines[:i]) + block + "".join(lines[i:])


class Submit:
    """Submit one level's displacement calculations and wait for them.

    Parameters
    ----------
    options : Options
    cma_level : str
        "A", "B" or "C"; the calculations live in ``<rootdir>/Disps<level>``.
    rootdir : str or Path
        Directory CMA was started in.
    prog_name : str
        molpro, psi4, orca or cfour.
    prog : str
        Program string from the options (may include "@version").
    """

    def __init__(self, options, cma_level, rootdir, prog_name, prog):
        self.cma_level = cma_level.upper()
        self.options = options
        self.prog = prog
        self.prog_name = prog_name
        self.rootdir = Path(rootdir)
        self.disps = self.rootdir / ("Disps" + self.cma_level)
        self.poll_interval = getattr(options, "poll_interval", 30)

    # ------------------------------------------------------------------
    def run(self, dirs=None):
        all_dirs = numbered_dirs(self.disps)
        if not all_dirs:
            raise CMAError(
                "No displacement directories found in {}.".format(self.disps)
            )
        dirs = all_dirs if dirs is None else sorted(int(d) for d in dirs)
        if not dirs:
            return

        script = build_job_script(
            self.options, self.cma_level, self.prog_name, self.prog, len(all_dirs)
        )
        cluster = self.options.cluster
        logger.info(
            "Submitting %d level-%s calculation(s) to %s.",
            len(dirs),
            self.cma_level,
            cluster,
        )

        if cluster == "slurm":
            if self.options.slurm_array:
                job_ids = self._submit_slurm_arrays(script, dirs)
            else:
                job_ids = self._submit_slurm_jobs(script, dirs)
            self._wait_slurm(job_ids)
        elif cluster == "sge":
            job_ids = self._submit_sge(script, dirs)
            self._wait_sge(job_ids)
        elif cluster == "custom":
            self._submit_custom(script, dirs)
            raise CMAWaiting(
                "Submitted {} level-{} job(s) with custom_submit_str. Run CMA again "
                "once they have finished.".format(len(dirs), self.cma_level),
                level=self.cma_level,
                missing=dirs,
            )
        else:
            raise CMAError(
                "cluster = {!r} does not submit jobs; this is a bug in the workflow.".format(
                    cluster
                )
            )

    # ------------------------------------------------------------------
    # Slurm
    # ------------------------------------------------------------------
    @staticmethod
    def plan_array_chunks(n_disps, max_array_size):
        """Split displacements 1..n into ``(offset, size)`` job-array chunks.

        Each chunk is submitted as ``--array=0-(size-1)`` with
        ``CMA_OFFSET=offset``, so array task ``i`` runs directory
        ``offset + i + 1``. Indices start at 0, so a chunk of
        ``max_array_size`` tasks stays within Slurm's MaxArraySize (the
        largest allowed index is MaxArraySize - 1).
        """
        return [
            (offset, len(task_ids))
            for offset, task_ids in Submit.array_blocks(
                range(1, n_disps + 1), max_array_size
            )
        ]

    @staticmethod
    def array_blocks(dirs, max_array_size):
        """Group directory numbers into ``(offset, task_ids)`` blocks whose
        task ids (``dir - 1 - offset``) all lie in ``0..max_array_size-1``."""
        max_array_size = int(max_array_size)
        if max_array_size < 1:
            raise CMAError("max_array_size must be at least 1")
        blocks = {}
        for d in sorted(dirs):
            offset = ((d - 1) // max_array_size) * max_array_size
            blocks.setdefault(offset, []).append(d - 1 - offset)
        return sorted(blocks.items())

    def _sbatch(self, args, cwd):
        job = subprocess.run(
            ["sbatch"] + args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        match = SBATCH_ID.search(job.stdout.decode("UTF-8", "replace"))
        if match is None:
            raise CMAError(
                "sbatch failed in {}:\n{}".format(
                    cwd, job.stderr.decode("UTF-8", "replace").strip()
                )
            )
        return match.group(1)

    def _submit_slurm_arrays(self, script, dirs):
        (self.disps / "sub_script.sh").write_text(
            insert_after_header(script, SLURM_ARRAY_PREAMBLE)
        )
        (self.disps / "slurm_logs").mkdir(exist_ok=True)
        blocks = self.array_blocks(dirs, self.options.max_array_size)
        logger.info(
            "%d displacement(s) -> %d job array(s) of at most %d tasks",
            len(dirs),
            len(blocks),
            self.options.max_array_size,
        )
        job_ids = []
        for offset, task_ids in blocks:
            spec = range_spec(task_ids)
            if self.options.array_throttle:
                spec += "%" + str(int(self.options.array_throttle))
            job_id = self._sbatch(
                [
                    "--array=" + spec,
                    "--export=ALL,CMA_OFFSET=" + str(offset),
                    "--output=slurm_logs/%x.%A_%a.out",
                    "--error=slurm_logs/%x.%A_%a.err",
                    "./sub_script.sh",
                ],
                self.disps,
            )
            first, last = offset + task_ids[0] + 1, offset + task_ids[-1] + 1
            logger.info(
                "Submitted array %s for displacements %d-%d", job_id, first, last
            )
            job_ids.append(job_id)
        return job_ids

    def _submit_slurm_jobs(self, script, dirs):
        job_ids = []
        for d in dirs:
            path = self.disps / str(d)
            (path / "sub_script.sh").write_text(script)
            job_ids.append(self._sbatch(["./sub_script.sh"], path))
            time.sleep(3)
        logger.info("Submitted %d Slurm job(s).", len(job_ids))
        return job_ids

    def _wait_slurm(self, job_ids):
        """Poll all outstanding jobs with one sacct call per interval. A job
        missing from sacct (accounting lag right after submission) only
        counts as finished after several misses."""
        pending = list(job_ids)
        misses = {j: 0 for j in pending}
        logger.info("Waiting for %d Slurm job(s) to finish...", len(pending))
        while pending:
            finish = subprocess.run(
                [
                    "sacct",
                    "-n",
                    "-X",
                    "-P",
                    "-o",
                    "JobID,State",
                    "-j",
                    ",".join(pending),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            listed, still_active = set(), set()
            for line in finish.stdout.decode("UTF-8", "replace").splitlines():
                fields = line.split("|")
                if len(fields) < 2:
                    continue
                base_id = fields[0].split("_")[0].split(".")[0]
                listed.add(base_id)
                if any(state in fields[1] for state in ACTIVE_SLURM_STATES):
                    still_active.add(base_id)
            remaining = []
            for job_id in pending:
                if job_id in still_active:
                    remaining.append(job_id)
                elif job_id not in listed and misses[job_id] < 3:
                    misses[job_id] += 1
                    remaining.append(job_id)
                else:
                    logger.debug("Slurm job %s is complete or failed", job_id)
            pending = remaining
            if pending:
                time.sleep(self.poll_interval)
        logger.info("All submitted Slurm jobs have left the queue.")
        time.sleep(15)  # let the file system catch up before reading outputs

    # ------------------------------------------------------------------
    # SGE
    # ------------------------------------------------------------------
    def _submit_sge(self, script, dirs):
        (self.disps / "displacements.sh").write_text(script)
        jobs = []
        for first, last in compress_ranges(dirs):
            proc = subprocess.run(
                ["qsub", "-t", "{}-{}".format(first, last), "displacements.sh"],
                cwd=str(self.disps),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            match = QSUB_ID.search(proc.stdout.decode("UTF-8", "replace"))
            if match is None:
                raise CMAError(
                    "qsub failed in {}:\n{}".format(
                        self.disps, proc.stderr.decode("UTF-8", "replace").strip()
                    )
                )
            jobs.append((match.group(1), last - first + 1))
            logger.info(
                "Submitted SGE task array %s for displacements %d-%d",
                match.group(1),
                first,
                last,
            )
        return jobs

    def _wait_sge(self, jobs):
        """qacct lists one 'taskid' record per finished task."""
        logger.info("Waiting for %d SGE task array(s) to finish...", len(jobs))
        pending = list(jobs)
        while pending:
            remaining = []
            for job_id, n_tasks in pending:
                proc = subprocess.run(
                    ["qacct", "-j", job_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                done = len(
                    re.findall(r"taskid", proc.stdout.decode("UTF-8", "replace"))
                )
                if done < n_tasks:
                    remaining.append((job_id, n_tasks))
            pending = remaining
            if pending:
                time.sleep(self.poll_interval)
        logger.info("All submitted SGE tasks have finished.")

    # ------------------------------------------------------------------
    # Custom
    # ------------------------------------------------------------------
    def _submit_custom(self, script, dirs):
        for d in dirs:
            path = self.disps / str(d)
            sub = path / "sub_script.sh"
            if (
                not sub.exists()
            ):  # a sub_script.sh in the run directory is copied in by DirectoryTree
                sub.write_text(script)
            command = "{} {}".format(self.options.custom_submit_str, sub.resolve())
            result = subprocess.run(command, shell=True, cwd=str(path))
            if result.returncode != 0:
                raise CMAError(
                    "custom_submit_str failed in {} (exit code {}): {}".format(
                        path, result.returncode, command
                    )
                )
            time.sleep(3)
