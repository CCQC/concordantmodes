import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class SapeloTemplate:
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
            "time_limit": options.time_limit,
            "memory": options.memory,
            "jarray": "1-{}".format(job_num),
            "prog_name": prog_name,
            "prog": prog,
            "tc": str(job_num),
            "cline": self.progdict[prog_name],
            "silly": "{}",
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

module load intel/2022a

# to change scratch dir to use local machine scratch
export SCRATCH_DIR=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
export APPTAINER_BIND="$SLURM_SUBMIT_DIR,$SCRATCH_DIR"

export TMPDIR=$SCRATCH_DIR

mpirun -n $NSLOTS apptainer exec /work/jttlab/containers/molpro-2021-gapr.sif molpro.exe input.dat --output $SLURM_SUBMIT_DIR/output.dat --nouse-logfile --directory $SCRATCH_DIR

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
#SBATCH --job-name=default            # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --constraint="EPYC|Intel"
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=4                    # Number of MPI ranks
#SBATCH --ntasks-per-node=4           # How many tasks on each node
#SBATCH --cpus-per-task=1             # Number of cores per MPI rank 
#SBATCH --mem=10G                     # total Memory no more per processor
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log
#SBATCH --mail-user="%u"@uga.edu
#SBTACH --mail-type=END,FAIL

cd $SLURM_SUBMIT_DIR
export NSLOTS=4
export THREADS=1  # can try adjusting but I don't think orca uses much omp threading if any

scratch_dir=/scratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $scratch_dir

module=ORCA/6.1.0-OpenMPI-4.1.8-GCC-13.3.0-avx2
module load $module
export OMP_NUM_THREADS=$THREADS

# Set other variables
base=`basename input.dat .dat`

# Copy Job/Executable Data
cp $SLURM_SUBMIT_DIR/input.dat $scratch_dir/input.dat
if [ -e $base.xyz ]; then cp $base.xyz $scratch_dir/guess.xyz ; fi
if [ -e $base.gbw ]; then cp $base.gbw $scratch_dir/guess.gbw ; fi
if [ -e $base.hess ]; then cp $base.hess $scratch_dir/guess.hess ; fi
if [ -e product.xyz ]; then cp product.xyz $scratch_dir/product.xyz ; fi
if [ -e ts_guess.xyz ]; then cp ts_guess.xyz $scratch_dir/ts_guess.xyz ; fi

echo " Running orca on `hostname`"
echo " Running calculation..."

cd $scratch_dir
/apps/eb/$module/bin/orca input.dat >& $SLURM_SUBMIT_DIR/output.dat || exit 1

echo " Saving data and cleaning up..."
# delete any temporary files that my be hanging around.
rm -f *.tmp*
find . -type f -size +50M -exec rm -f {silly} \;
tar --exclude='*tmp*' --transform "s,^,Job_Data_$SLURM_JOB_ID/," -vzcf $SLURM_SUBMIT_DIR/Job_Data_$SLURM_JOB_ID.tar.gz *

echo " Job complete on `hostname`."

rm $scratch_dir -r

"""

    def run(self):
        return self.sapelo_template.format(**self.odict)
