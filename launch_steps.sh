startdate="202405121430"
enddate="202405122000"
interval=10
for timestamp in `./daterange.sh $startdate $enddate $interval`
do
  echo Launching: qsub -vtimestamp=$timestamp jobscript_steps.sh 
  qsub -vtimestamp=$timestamp jobscript_steps.sh
done


startdate="202406271400"
enddate="202406272000"
for timestamp in `./daterange.sh $startdate $enddate $interval`
do
  echo Launching: qsub -vtimestamp=$timestamp jobscript_steps.sh 
  qsub -vtimestamp=$timestamp jobscript_steps.sh
done
