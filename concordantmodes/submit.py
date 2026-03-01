import re
import shutil
import subprocess
import time
import os

from subprocess import Popen

from concordantmodes.vulcan_template import VulcanTemplate
from concordantmodes.sapelo_template import SapeloTemplate


class Submit(object):
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
            os.chdir("..")

            print(
                "Jobs have been submitted. You will need to come back when they finish and run CMA again with relevent gen_disps and calc keywords set to false."
            )
            raise RuntimeError

        else:
            print(
                "Only Vulcan, Sapelo, or Custom cluster options are available, select one of those!"
            )
            raise RuntimeError
