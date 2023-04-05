import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA


class VulcanTemplate(object):
    """
    This file just stores the vulcan qsub script
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
        }
        # This can be inserted back in if the sync keyword is sorted
        # $ -sync y
        if self.prog_name == "cfour":
            self.vulcan_template = """#!/bin/sh
#$ -q {q}
#$ -N Concordant
#$ -S /bin/sh
#$ -cwd
#$ -t {jarray}
#$ -tc {tc}

. /etc/profile.d/modules.sh

# Disable production of core dump files
ulimit -c 0

echo ""
echo "***********************************************************************"
echo " Starting job:"
echo ""
echo "    Name:              "$JOB_NAME
echo "    ID:                "$JOB_ID
echo "    Hostname:          "$HOSTNAME
echo "    Working directory: "$SGE_O_WORKDIR
echo ""
echo "    Submitted using:   Concordant Modes Approach "
echo "***********************************************************************"

# cd into individual task directory
cd $SGE_O_WORKDIR/$SGE_TASK_ID
vulcan load {cline}

scratch=$TMPDIR/$USER/$JOB_ID

# Set variables
export OMP_NUM_THREADS=4
export NSLOTS=1
prefix=/opt/vulcan/opt/vulcan/linux-x86_64/intel-13.0.0/cfour-2.0-yhj426etc3g7hslvbmpgvdymp2w76rob

# Copy job data
cp $SGE_O_WORKDIR/$SGE_TASK_ID/ZMAT $scratch/ZMAT
cp $prefix/basis/GENBAS $scratch
cp $prefix/basis/ECPDATA $scratch
if [ -e JAINDX ]; then cp JAINDX $scratch ; fi
if [ -e JOBARC ]; then cp JOBARC $scratch ; fi
if [ -e FCMINT ]; then cp FCMINT $scratch ; fi
if [ -e GENBAS ]; then cp GENBAS $scratch ; fi
if [ -e ECPDATA ]; then cp ECPDATA $scratch ; fi
if [ -e OPTARC ]; then cp OPTARC $scratch ; fi
if [ -e ISOTOPES ]; then cp ISOTOPES $scratch ; fi
if [ -e ISOMASS ]; then cp ISOMASS $scratch ; fi
if [ -e initden.dat ]; then cp initden.dat $scratch ; fi
if [ -e OLDMOS ]; then cp OLDMOS $scratch ; fi

echo " Running cfour on `hostname`"
echo " Running calculation..."

cd $scratch
xcfour >& $SGE_O_WORKDIR/$SGE_TASK_ID/output.dat
xja2fja
/opt/scripts/cfour2avogadro $SGE_O_WORKDIR/$SGE_TASK_ID/output.dat

echo " Saving data and cleaning up..."
if [ -e ZMATnew ]; then cp -f ZMATnew $SGE_O_WORKDIR/$SGE_TASK_ID/ZMATnew ; fi

# Create a job data archive file
tar --transform "s,^,Job_Data_$JOB_ID/," -vcf $SGE_O_WORKDIR/$SGE_TASK_ID/Job_Data_$JOB_ID.tar OPTARC FCMINT FCMFINAL ZMATnew JMOL.plot JOBARC JAINDX FJOBARC DIPDER HESSIAN MOLDEN NEWMOS AVOGADROplot.log den.dat
if [ -e zmat001 ]; then tar --transform "s,^,Job_Data_$JOB_ID/," -vrf $SGE_O_WORKDIR/$SGE_TASK_ID/Job_Data_$JOB_ID.tar zmat* ; fi
gzip $SGE_O_WORKDIR/$SGE_TASK_ID/Job_Data_$JOB_ID.tar

echo " Job complete on `hostname`." """
        else:
            self.vulcan_template = """#!/bin/sh
#$ -q {q}
#$ -N Concordant
#$ -S /bin/sh
#$ -cwd
#$ -t {jarray}
#$ -tc {tc}

. /etc/profile.d/modules.sh

# Disable production of core dump files
ulimit -c 0

echo ""
echo "***********************************************************************"
echo " Starting job:"
echo ""
echo "    Name:              "$JOB_NAME
echo "    ID:                "$JOB_ID
echo "    Hostname:          "$HOSTNAME
echo "    Working directory: "$SGE_O_WORKDIR
echo ""
echo "    Submitted using:   Concordant Modes Approach "
echo "***********************************************************************"

# cd into individual task directory
cd $SGE_O_WORKDIR/$SGE_TASK_ID
vulcan load {prog}

export NSLOTS={nslots}

{cline}"""

    def run(self):
        return self.vulcan_template.format(**self.odict)
