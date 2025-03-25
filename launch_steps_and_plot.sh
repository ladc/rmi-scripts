startdate="202308250230"
enddate="202308251900"
interval=30
jobid="-1"

set -x

qsub -vtimestamp=202308250400,nfiles=48 jobscript_plot_obs.sh
qsub -vtimestamp=202308250800,nfiles=48 jobscript_plot_obs.sh
qsub -vtimestamp=202308251200,nfiles=48 jobscript_plot_obs.sh

for timestamp in `./daterange.sh $startdate $enddate $interval`
do
  echo Launching: qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_steps.sh 
  jobid=`qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_steps.sh`
  qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_plot.sh
done


