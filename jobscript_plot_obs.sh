#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=0:59:00
#PBS -l select=1:ncpus=1:ompthreads=1:mem=24GB
#PBS -j oe
#PBS -W umask=022
#PBS -N plot_obs 
echo -----------------------------------
echo START : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------

# Run this script as follows:
# qsub -vtimestamp=YYYYMMDDHHMM,nfiles=24 jobscript_plot_obs.sh

#set -x # echo script lines as they are executed
#set -e # stop the shell on first error
#set -u # fail when using an undefined variable

source $HOME/.bashrc
export OMP_NUM_THREADS=1

workdir=$HOME/projects/rmi-scripts
logdir=${workdir}/data/log
logfile=${logdir}/obsplot_${timestamp}
script=plot_obs.py
conda init
conda activate pysteps_dev
cd $workdir
mkdir -p $logdir

set -x
python ${script} ${timestamp} ${nfiles} > ${logfile}.log 2> ${logfile}.err
set +x

echo -----------------------------------
echo STOP : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------


