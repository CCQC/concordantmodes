#!/bin/sh
#$ -q gen4.q,gen6.q,debug.q
#$ -N Concordant
#$ -S /bin/sh
#$ -cwd
#$ -t 1-9
#$ -tc 9

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
vulcan load molpro@2010.1.67+mpi

export NSLOTS=1

molpro -n $NSLOTS --nouse-logfile -o                 output.dat input.dat