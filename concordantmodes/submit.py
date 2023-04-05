import re
import subprocess
import time
import os
from subprocess import Popen


class Submit(object):
    def __init__(self, disp_list, options):
        self.disp_list = disp_list
        self.options = options

    def run(self):
        if self.options.cluster != "sapelo":
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
                if len(job_match) == len(self.disp_list):
                    break
                time.sleep(10)

            output = str(process.stdout)
            error = str(process.stderr)
            pass

        else:
            from subprocess import Popen

            processes = []
            print(os.getcwd())
            for z in range(len(self.disp_list)):
                # path = 'Disps' + '/' +  str(z + 1) + '/'
                path = str(z + 1) + "/"
                pipe = subprocess.PIPE
                # process = Popen(['sh',  './sub.sh'], cwd = path, stdout=pipe, stderr=pipe, shell = True)
                job = subprocess.run(
                    ["sbatch", "./optstep.sh"], cwd=path, stdout=pipe, stderr=pipe
                )
                processes.append(job)
                time.sleep(2)

            for q in range(len(processes)):
                while True:
                    job = processes[q]
                    outRegex = r"Submitted\s*batch\s*job(?:-array)?\s*(\d*)"
                    job_id = int(
                        re.search(outRegex, job.stdout.decode("UTF-8")).group(1)
                    )
                    jobFinRegex = re.compile(r"taskid")
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
            print("sleeping")
            time.sleep(10)
