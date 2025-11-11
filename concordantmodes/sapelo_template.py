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
            "orca": "",
        }
        self.odict = {
            "q": options.queue,
            "nslots": options.nslots,
            "time_limit" : options.time_limit,
            "memory" : options.memory,
            "jarray": "1-{}".format(job_num),
            "prog_name": prog_name,
            "prog": prog,
            "tc": str(job_num),
            "cline": self.progdict[prog_name],
        }
        # This can be inserted back in if the sync keyword is sorted
        # $ -sync y
        if self.prog_name == "molpro":
            self.sapelo_template = """#!/bin/sh
#!/bin/bash
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --constraint=EPYC|Intel
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=4             # Number of MPI ranks
#SBATCH --ntasks-per-node=4    # How many tasks on each node
#SBATCH --cpus-per-task=1     # Number of cores per MPI rank 
#SBATCH --mem=32GB        # Memory per processor
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=4
export THREADS=1

module load intel/2023a

# to change scratch dir to use local machine scratch
export SCRATCH_DIR=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
export APPTAINER_BIND="$SLURM_SUBMIT_DIR,$SCRATCH_DIR"

export TMPDIR=$SCRATCH_DIR

mpirun -n $NSLOTS apptainer exec /work/jttlab/containers/molpro_mpipr.sif molpro.exe input.dat --output $SLURM_SUBMIT_DIR/output.dat --nouse-logfile --directory $SCRATCH_DIR

rm $SCRATCH_DIR -r


#ignored line -- do not remove
"""
        elif self.prog_name == "psi4":
            self.sapelo_template = """#!/bin/sh
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --constraint=EPYC|Intel
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=4             # Number of MPI ranks
#SBATCH --ntasks-per-node=4    # How many tasks on each node
#SBATCH --cpus-per-task=1     # Number of cores per MPI rank 
#SBATCH --mem=32GB        # Memory per processor
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=4
export THREADS=1

set -eE
trap 'cleanup' EXIT

function cleanup(){{
  echo "Exiting. Performing Cleanup"
  rm $PSI_SCRATCH -r
}}

export PSI_SCRATCH=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $PSI_SCRATCH
psi4 -n $NSLOTS -o output.dat

#ignored line -- do not remove
"""
# rm $PSI_SCRATCH -r
        elif self.prog_name == "orca":
            self.sapelo_template = """#!/bin/bash
#SBATCH --job-name=Concordant           # Job name (testBowtie2)
#SBATCH --partition=batch               # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --ntasks=10                     # 1 task (process) for below commands
#SBATCH --cpus-per-task=1               # CPU core count per task, by default 1 CPU core per task
#SBATCH --mem-per-cpu=10G
#SBATCH --time={time_limit}
#SBATCH --output=%x_%j.out              # Standard output log, e.g., testBowtie2_12345.out

ml OpenMPI/4.1.4-GCC-11.3.0
ml ORCA/6.1.0-OpenMPI-4.1.8-GCC-13.3.0-avx2
/apps/eb/ORCA/6.1.0-OpenMPI-4.1.8-GCC-13.3.0-avx2/bin/orca input.dat > output.dat
"""

    def run(self):
        return self.sapelo_template.format(**self.odict)
