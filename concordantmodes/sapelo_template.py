import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class SapeloTemplate(object):
    """
    This file just stores the sapelo optstep script
    """

    def __init__(self, options, job_num, prog_name, prog):
        self.prog_name = prog_name
        self.progdict = {
            "molpro": "molpro -n $NSLOTS --nouse-logfile --no-xml-output -o \
                output.dat input.dat",
            "psi4": "psi4 -n $NSLOTS",
            "cfour": prog + "+vectorization",
        }
        self.odict = {
            "q": options.queue,
            "nslots": options.nslots,
            "jarray": "1-{}".format(job_num),
            "prog_name": prog_name,
            "prog": prog,
            "tc": str(job_num),
            "cline": self.progdict[prog_name],
            # "job_name": options.job_name,
            # "job_name": options.job_name,
        }
        # This can be inserted back in if the sync keyword is sorted
        # $ -sync y
        if self.prog_name != "cfour":
            self.sapelo_template = """#!/bin/sh
#!/bin/bash
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --constraint=Intel
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=6             # Number of MPI ranks
#SBATCH --ntasks-per-node=6    # How many tasks on each node
#SBATCH --cpus-per-task=1     # Number of cores per MPI rank 
#SBATCH --mem=60GB        # Memory per processor
#SBATCH --time=72:00:00
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=6
export THREADS=1

module load intel/2019b
export PATH=$PATH:/work/jttlab/molpro/2010/bin/

scratch_dir=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $scrath_dir

time molpro -n $NSLOTS --nouse-logfile --no-xml-output --output output.dat --directory $scratch_dir input.dat

tar -cvzf molpro_tmp.tar.gz $scratch_dir/*
rm $scratch_dir


#ignored line -- do not remove
"""

    def run(self):
        return self.sapelo_template.format(**self.odict)
