import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class SisyphusTemplate:
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
            "xtb" : "input.coord",
            "gxtb" : "input.coord",
        }
        self.odict = {
            "q": options.queue,
            "nslots": options.nslots,
            "jarray": "1-{}".format(job_num),
            "prog_name": prog_name,
            "prog": prog,
            "tc": str(job_num),
            "cline": self.progdict[prog_name],
        }
        # This can be inserted back in if the sync keyword is sorted
        # $ -sync y
        if self.prog_name == "psi4":
            self.sisyphus_template = """#!/bin/bash
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --cpus-per-task=4     # Number of cores per MPI rank 
#SBATCH --mem=40GB        # Memory per processor
#SBATCH --time=72:00:00
#SBATCH --output=output.dat
#SBATCH --hint=compute_bound

source /nfs/cluster_config/modules.sh
module load psi4/nightly

eval "$(conda shell.bash hook)"
# Load their personal Conda environment
conda activate p4dev
which psi4
#capture the submission directory
SUBMIT_DIR="$SLURM_SUBMIT_DIR"

#scratch directory
SCRATCH_DIR="/scratch/$USER/$SLURM_JOB_ID"

srun --cpu-bind=verbose psi4 -i input.dat -o output.dat


echo "Job \$SLURM_JOB_ID running on \$HOSTNAME"
echo "CPU affinity:"
taskset -cp $$
"""

        elif self.prog_name == "molpro":
            self.sisyphus_template = """#!/bin/bash
#SBATCH --job-name=STEP--00-1                 # Job name
#SBATCH --partition=batch                   # Partition (queue) name
#SBATCH --ntasks-per-node=8    # How many tasks on each node
#SBATCH --cpus-per-task=1     # Number of cores per MPI rank 
#SBATCH --mem=85GB                        # Total memory
#SBATCH --time=10:00:00
#SBATCH --output="%x.%j".out              # Standard output log
#SBATCH --error="%x.%j".err               # Standard error log

# Load Molpro
module purge
module load molpro

#Run Molpro

export SCRATCH_DIR=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR

# Run Molpro
molpro -n $SLURM_NTASKS input.dat --output output.dat --directory $SCRATCH_DIR

#ignored line --do not remove
"""

        elif self.prog_name == "gxtb":
            self.sisyphus_template = """#!/bin/bash
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --cpus-per-task=4     # Number of cores per MPI rank 
#SBATCH --mem=40GB        # Memory per processor
#SBATCH --time=72:00:00
#SBATCH --output=output.dat
#SBATCH --hint=compute_bound

#capture the submission directory
SUBMIT_DIR="$SLURM_SUBMIT_DIR"

#scratch directory
SCRATCH_DIR="/scratch/$USER/$SLURM_JOB_ID"

srun --cpu-bind=verbose gxtb -c input.coord


echo "Job \$SLURM_JOB_ID running on \$HOSTNAME"
echo "CPU affinity:"
taskset -cp $$
"""
        elif self.prog_name == "xtb":
            self.sisyphus_template = """#!/bin/bash
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --cpus-per-task=4     # Number of cores per MPI rank 
#SBATCH --mem=40GB        # Memory per processor
#SBATCH --time=72:00:00
#SBATCH --output=output.dat
#SBATCH --hint=compute_bound

source /nfs/cluster_config/modules.sh
module load xtb/6.7.1

#eval "$(conda shell.bash hook)"
## Load their personal Conda environment
#conda activate xtbenv
#capture the submission directory
SUBMIT_DIR="$SLURM_SUBMIT_DIR"

#scratch directory
SCRATCH_DIR="/scratch/$USER/$SLURM_JOB_ID"

srun --cpu-bind=verbose {xtb_variant} {cline} {xtb_addtl_opts}


echo "Job \$SLURM_JOB_ID running on \$HOSTNAME"
echo "CPU affinity:"
taskset -cp $$
"""

    def run(self):
        return self.sisyphus_template.format(**self.odict)
