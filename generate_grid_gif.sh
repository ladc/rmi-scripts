# Usage: pass timestamp.

timestamp=$1

echo "Generating animation for $timestamp"

figdir="data/figs_$timestamp"
tmpdir="data/figs_$timestamp/tmp"

obsdir="data/figs/observations"

radar_interval_s=300 # 300s or 5 min
hist=12 # Plot past hist radar images before the nowcast.
max_lt=72 # Max leadtime to be plotted.

timestamp_date=$(date -u -d "${timestamp:0:4}-${timestamp:4:2}-${timestamp:6:2} ${timestamp:8:2}:${timestamp:10:2}" +"%s")

echo $timestamp_date

first_date=$((timestamp_date - hist * radar_interval_s))
first_timestamp=$(date -u -d "@$first_date" +"%Y%m%d%H%M")

mkdir -p $tmpdir

for stamp in `./daterange.sh $first_timestamp $timestamp 5`
do
        f=${obsdir}/steps_radar_${stamp}.png
	montage $f $f $f $f  -tile 2x2 -geometry +0+0 -background none ${tmpdir}/0_`basename $f` &
done

wait 
for i in `seq -w 1 $max_lt`;
do
	montage ${figdir}/steps_pr01/steps_pr01_ac05A0+${i}.png ${figdir}/steps_en00/steps_en00_ac05A0+${i}.png ${figdir}/steps_mean/steps_mean_ac05A0+${i}.png ${figdir}/steps_radar/steps_radar_ac05A0+${i}.png -tile 2x2 -geometry +0+0 -background none ${tmpdir}/1_${i}.png &
        if ((i % 8 == 0)); then wait; fi
done

wait
convert -delay 50 ${tmpdir}/*.png -loop 0 $figdir/pr_ens_mean_obs_${timestamp}.gif


