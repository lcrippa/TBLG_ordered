#!/bin/bash
#SBATCH --ntasks=80  # should be a multiple of 8
#SBATCH --time=12:00:00
#SBATCH --export=NONE

source /sw/batch/init.sh
source /home/bbf3609/.bashrc

module switch env env/oneapi-2024.2.1_impi

NCPU=80
BETA=1.0
NLOOP=20
LOGSTEPS=6
BIAS=0.1
MIX=0.5
SYM="K-IVC"

conda activate triqs

mpirun --tag-output -np $NCPU python3 m2.py $BETA $NLOOP $LOGSTEPS $BIAS --mix $MIX --sym $SYM > LOG.out 2>&1

exit
