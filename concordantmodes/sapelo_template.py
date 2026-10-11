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
            "psi4": "psi4 -n $THREADS",
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
#SBATCH --mem=120GB        # Memory per processor
#SBATCH --gres=lscratch:100
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=4
export THREADS=1

module load intel/2022a

# to change scratch dir to use local machine scratch
export SCRATCH_DIR=/lscratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
export APPTAINER_BIND="$SLURM_SUBMIT_DIR,$SCRATCH_DIR"

export TMPDIR=$SCRATCH_DIR

mpirun -n $NSLOTS apptainer exec /work/hfslab/containers/molpro-2021-gapr.sif molpro.exe input.dat --output $SLURM_SUBMIT_DIR/output.dat --nouse-logfile --directory $SCRATCH_DIR

rm $SCRATCH_DIR -r


#ignored line -- do not remove
"""
        # SBATCH --ntasks-per-node=1    # How many tasks on each node
        # SBATCH --nodes=1                     # Number of nodes
        # export NSLOTS=1
        elif self.prog_name == "psi4":
            self.sapelo_template = """#!/bin/sh
#SBATCH --job-name=Concordant             # Job name
#SBATCH --partition=batch               # Partition (queue) name
#SBATCH --constraint=EPYC|Intel
#SBATCH --ntasks=1             # Number of MPI ranks
#SBATCH --cpus-per-task=8     # Number of cores per MPI rank 
#SBATCH --mem=120GB        # Memory per processor
#SBATCH --gres=lscratch:100
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log

cd $SLURM_SUBMIT_DIR
export THREADS=8

set -eE
trap 'cleanup' EXIT

function cleanup(){{
  echo "Exiting. Performing Cleanup"
  rm $PSI_SCRATCH -r
}}
source ~/.bashrc
conda activate psi4_dlpno
export PSI_SCRATCH=/lscratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $PSI_SCRATCH
psi4 -n $THREADS -o output.dat

#ignored line -- do not remove
"""
        # psi4 -n $NSLOTS -o output.dat
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
#SBATCH --mem=120G                     # total Memory no more per processor
#SBATCH --gres=lscratch:100
#SBATCH --time={time_limit}
#SBATCH --output="%x.%j".out     # Standard output log
#SBATCH --error="%x.%j".err      # Standard error log
#SBATCH --mail-user="%u"@uga.edu
#SBTACH --mail-type=END,FAIL

cd $SLURM_SUBMIT_DIR
export NSLOTS=4
export THREADS=1  # can try adjusting but I don't think orca uses much omp threading if any

scratch_dir=/lscratch/$USER/tmp/$SLURM_JOB_ID
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
        elif self.prog_name == "cfour":
            self.sapelo_template = """#!/bin/bash
#SBATCH --partition=batch
#SBATCH --constraint="EPYC|Intel"
#SBATCH --job-name=azulene-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=60GB
#SBATCH --gres=lscratch:100
#SBATCH --output="%x.%j".out    # Standard output log
#SBATCH --error="%x.%j".err     # Standard error log

cd $SLURM_SUBMIT_DIR
export NSLOTS=1

module=cfour/2.1-intel-2023a-serial
export OMP_NUM_THREADS=$NSLOTS

scratch_dir=/lscratch/$USER/tmp/$SLURM_JOB_ID
mkdir -p $scratch_dir


# make sure MRCC is around just in case
export PATH=$PATH:/work/hfslab/mrcc/2020/
prefix=/apps/eb/$module/
module load $module

# Copy job data
if [[ -e input.dat && ! -e ZMAT ]]; then
  cp input.dat $scratch_dir/ZMAT
else
  cp $SLURM_SUBMIT_DIR/ZMAT $scratch_dir
fi

cp $prefix/basis/GENBAS $scratch_dir
cp $prefix/basis/ECPDATA $scratch_dir
if [ -e JAINDX ]; then cp JAINDX $scratch_dir ; fi
if [ -e JOBARC ]; then cp JOBARC $scratch_dir ; fi
if [ -e FCMINT ]; then cp FCMINT $scratch_dir ; fi
if [ -e GENBAS ]; then cp GENBAS $scratch_dir ; fi
if [ -e ECPDATA ]; then cp ECPDATA $scratch_dir ; fi
if [ -e OPTARC ]; then cp OPTARC $scratch_dir ; fi
if [ -e ISOTOPES ]; then cp ISOTOPES $scratch_dir ; fi
if [ -e ISOMASS ]; then cp ISOMASS $scratch_dir ; fi
if [ -e initden.dat ]; then cp initden.dat $scratch_dir ; fi
if [ -e OLDMOS ]; then cp OLDMOS $scratch_dir ; fi

echo " Running cfour on `hostname`"
echo " Running calculation..."

cd $scratch_dir
xcfour >& $SLURM_SUBMIT_DIR/output.dat
xja2fja

echo " Saving data and cleaning up..."
if [ -e ZMATnew ]; then cp -f ZMATnew $SLURM_SUBMIT_DIR/ZMATnew ; fi
if [ -e GRD ]; then cp -f GRD $SLURM_SUBMIT_DIR/GRD ; fi
if [ -e FCMFINAL ]; then cp -f FCMFINAL $SLURM_SUBMIT_DIR/FCMFINAL ; fi

# Create a job data archive file
tar --transform "s,^,Job_Data_$SLURM_JOB_ID/," -vcf $SLURM_SUBMIT_DIR/Job_Data_$SLURM_JOB_ID.tar OPTARC FCMINT FCMFINAL ZMATnew JMOL.plot JOBARC JAINDX FJOBARC DIPDER HESSIAN MOLDEN NEWMOS den.dat
if [ -e zmat001 ]; then tar --transform "s,^,Job_Data_$SLURM_JOB_ID/," -vrf $SLURM_SUBMIT_DIR/Job_Data_$SLURM_JOB_ID.tar zmat* ; fi
gzip $SLURM_SUBMIT_DIR/Job_Data_$SLURM_JOB_ID.tar

echo " Job complete on `hostname`."

rm $scratch_dir -r

# ignored line -- do not remove
"""

    def run(self):
        return self.sapelo_template.format(**self.odict)
