startdate="202409071700"
enddate="202409072200"
interval=30
jobid=-1

for timestamp in `./daterange.sh $startdate $enddate $interval`
do
  echo Launching: qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_steps.sh 
  jobid=`qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_steps.sh`
  qsub -vtimestamp=$timestamp -W depend=afterok:$jobid jobscript_plot.sh
done
