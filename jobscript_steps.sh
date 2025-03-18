#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=0:59:00
#PBS -l select=1:ncpus=8:ompthreads=8:mem=24GB
#PBS -j oe
#PBS -o /home/ledecruz/projects/rmi-scripts/data/log/steps.log
#PBS -e /home/ledecruz/projects/rmi-scripts/data/log/steps.err
#PBS -W umask=022
#PBS -N pysteps_run
echo -----------------------------------
echo START : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------

set -x # echo script lines as they are executed
#set -e # stop the shell on first error
#set -u # fail when using an undefined variable
source /home/ledecruz/.bashrc
export OMP_NUM_THREADS=8
workdir=/home/ledecruz/projects/rmi-scripts
script=run_steps.py
#script=plot_nwc.py
conda init
conda activate pysteps_dev
cd $workdir

python ${script} $timestamp 72 8 8 > $workdir/data/log/steps_$timestamp.log 2> $workdir/data/log/steps_$timestamp.err


echo -----------------------------------
echo STOP : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------


