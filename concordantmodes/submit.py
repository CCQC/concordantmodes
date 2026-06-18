import re
import shutil
import subprocess
import time
import os

from subprocess import Popen

from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.sapelo_template import SapeloTemplate
from concordantmodes.sisyphus_template import SisyphusTemplate


class Submit:
    """
    Submit and monitor batches of Concordant Modes displacement calculations on
    supported computing clusters.

    This class automates the execution of electronic structure calculations
    associated with generated displacement geometries by creating appropriate
    scheduler submission scripts, launching jobs, and optionally monitoring
    their completion. Supported execution environments include Sun Grid Engine
    (SGE), Slurm, and user-defined custom submission workflows.

    Depending on the selected cluster type, the class will:

    * Generate scheduler-specific submission scripts using predefined
      templates.
    * Submit displacement calculations as either job arrays (SGE) or
      individual jobs (Slurm).
    * Monitor submitted jobs until completion when supported.
    * Support externally managed submission workflows through a user-supplied
      command string.
    * Return control to the CMA workflow once all displacement calculations
      have completed successfully.

    Parameters
    ----------
    options : Options
        Program options object containing cluster configuration, scheduler
        settings, submission commands, and resource specifications.
    cma_level : str
        Concordant Modes displacement level being processed ("A", "B", or
        "C"). Used to locate the corresponding displacement directory.
    rootdir : str
        Root working directory containing the displacement calculation
        directories.
    prog_name : str
        Name of the electronic structure package being executed (e.g.
        Molpro, Psi4, CFOUR, ORCA).
    prog : str
        Program execution command or executable path used by the submission
        templates.

    Attributes
    ----------
    cma_level : str
        Current CMA displacement level.
    options : Options
        Runtime options controlling submission behavior.
    prog : str
        Electronic structure program executable or launch command.
    prog_name : str
        Electronic structure package name.
    rootdir : str
        Root directory containing displacement calculations.

    Notes
    -----
    The class assumes that displacement directories have already been created
    and numbered sequentially beginning with directory ``1``.

    Scheduler-specific behavior:

    **SGE**
        Generates a job-array submission script using ``VulcanTemplate``,
        submits it with ``qsub``, and monitors completion through
        ``qacct`` records.

    **Slurm**
        Generates a submission script using ``SapeloTemplate``, copies the
        script into each displacement directory, submits jobs using
        ``sbatch``, and polls job status through ``sacct``.

    **Custom**
        Executes a user-defined submission command specified by
        ``options.custom_submit_str``. Because job completion cannot be
        monitored generically, execution terminates after submission and
        requires the user to rerun CMA after the jobs finish.

    Raises
    ------
    RuntimeError
        If an unsupported cluster type is specified.
    RuntimeError
        If the custom submission option is selected without a valid
        submission command.
    RuntimeError
        After custom job submission to intentionally terminate execution
        until external calculations have completed.

    See Also
    --------
    VulcanTemplate
        Generates SGE submission scripts.
    SapeloTemplate
        Generates Slurm submission scripts.

    Examples
    --------
    Submit displacement calculations on a Slurm cluster::

        submitter = Submit(
            options,
            cma_level="B",
            rootdir=workdir,
            prog_name="molpro",
            prog="molpro"
        )
        submitter.run()

    Submit calculations using a custom scheduler wrapper::

        options.cluster = "custom"
        options.custom_submit_str = "my_submit_command"

        submitter = Submit(
            options,
            cma_level="A",
            rootdir=workdir,
            prog_name="psi4",
            prog="psi4"
        )
        submitter.run()
    """

    def __init__(self, options, cma_level, rootdir, prog_name, prog):
        self.cma_level = cma_level
        self.options = options
        self.prog = prog
        self.prog_name = prog_name
        self.rootdir = rootdir

    def run(self):
        disp_list = []

        for i in os.listdir(self.rootdir + "/Disps" + self.cma_level):
            disp_list.append(i)

        os.chdir(self.rootdir + "/Disps" + self.cma_level)

        # TODO move Vulcan and Sapelo templates to more general sge and slurm templates.
        if self.options.cluster.lower() == "sge":
            v_template = VulcanTemplate(
                self.options, len(disp_list), self.prog_name, self.prog
            )
            out = v_template.run()

            with open("displacements.sh", "w") as file:
                file.write(out)

            pipe = subprocess.PIPE

            process = subprocess.run(
                "qsub displacements.sh", stdout=pipe, stderr=pipe, shell=True
            )
            self.out_regex = re.compile(r"Your\s*job\-array\s*(\d*)")
            self.job_id = int(re.search(self.out_regex, str(process.stdout)).group(1))

            self.job_fin_regex = re.compile(r"taskid")
            while True:
                qacct_proc = subprocess.run(
                    ["qacct", "-j", str(self.job_id)], stdout=pipe, stderr=pipe
                )
                qacct_string = str(qacct_proc.stdout)
                job_match = re.findall(self.job_fin_regex, qacct_string)
                if len(job_match) == len(disp_list):
                    break
                time.sleep(30)

            output = str(process.stdout)
            error = str(process.stderr)
            pass

        elif self.options.cluster.lower() == "slurm":
            s_template = SapeloTemplate(
                self.options, len(disp_list), self.prog_name, self.prog
            )
            #s_template = SisyphusTemplate(
            #    self.options, len(disp_list), self.prog_name, self.prog
            #)
            out = s_template.run()

            with open("sub_script.sh", "w") as file:
                file.write(out)

            for z in range(len(disp_list)):
                source = os.getcwd() + "/sub_script.sh"
                os.chdir("./" + str(z + 1))
                destination = os.getcwd()
                shutil.copy2(source, destination)
                os.chdir("../")

            processes = []

            for z in range(len(disp_list)):
                path = str(z + 1) + "/"
                pipe = subprocess.PIPE
                job = subprocess.run(
                    ["sbatch", "./sub_script.sh"], cwd=path, stdout=pipe, stderr=pipe
                )
                processes.append(job)
                time.sleep(3)

            for q in range(len(processes)):
                while True:
                    job = processes[q]
                    outRegex = r"Submitted\s*batch\s*job(?:-array)?\s*(\d*)"
                    job_id = int(
                        re.search(outRegex, job.stdout.decode("UTF-8")).group(1)
                    )
                    finish = subprocess.run(
                        ["sacct", "-j", str(job_id)], stdout=pipe, stderr=pipe
                    )
                    output = str(finish.stdout.decode("UTF-8"))
                    if not ("PENDING" in output or "RUNNING" in output):
                        print(
                            "job id "
                            + str(job_id)
                            + " must be complete or failed "
                            + str(q)
                        )
                        break
            print("Napping")
            time.sleep(15)
        elif self.options.cluster.lower() == "custom":
            # Here we move into the disp directory then execute
            # a command line argument specified by the user.

            if not len(self.options.custom_submit_str):
                print(
                    "The custom_submit_str option cannot be empty when using this option."
                )
                raise RuntimeError

            for z in range(len(disp_list)):
                path = str(z + 1) + "/"
                os.chdir(path)
                os.system(
                    self.options.custom_submit_str
                    + " "
                    + os.getcwd()
                    + "/sub_script.sh"
                )
                os.chdir("..")
                time.sleep(3)
            # os.chdir("..")

            print(
                "Jobs have been submitted. You will need to come back when they finish and run CMA again with relevent gen_disps and calc keywords set to false."
            )
            raise RuntimeError

        else:
            print(
                "Only Vulcan, Sapelo, or Custom cluster options are available, select one of those!"
            )
            raise RuntimeError
        os.chdir("../")
