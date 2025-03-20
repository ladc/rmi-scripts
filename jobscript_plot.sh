#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=0:59:00
#PBS -l select=1:ncpus=8:ompthreads=8:mem=24GB
#PBS -j oe
#PBS -W umask=022
#PBS -N pysteps_plot

# Run this script as follows:
# qsub -vtimestamp=202308251600 jobscript_plot.sh 

echo -----------------------------------
echo START : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------

#set -x # echo script lines as they are executed
#set -e # stop the shell on first error
#set -u # fail when using an undefined variable
source /home/ledecruz/.bashrc
export OMP_NUM_THREADS=8
workdir=/home/ledecruz/projects/rmi-scripts
logdir=${workdir}/data/log
logfile=${logdir}/plot_${timestamp}
script=plot_nwc.py
conda init
conda activate pysteps_dev

set -x
cd $workdir
python ${script} $timestamp > ${logfile}.log 2> ${logfile}.err
set +x

module load ImageMagick

set -x
./generate_grid_gif.sh $timestamp >> ${logfile}.log 2>> ${logfile}.err
set +x

echo -----------------------------------
echo STOP : `date -u +"%Y%m%d %H:%M:%S"`
echo -----------------------------------


