figdir="../figs"

max_lt=18

mkdir -p tmp

for f in ${figdir}/observations/*.png
do
	montage $f $f $f $f  -tile 2x2 -geometry +0+0 -background none tmp/0_`basename $f` &
done

wait 
for i in `seq -w 1 $max_lt`;
do
	montage ${figdir}/steps_pr01/steps_pr01_ac05A0+${i}.png ${figdir}/steps_en00/steps_en00_ac05A0+${i}.png ${figdir}/steps_mean/steps_mean_ac05A0+${i}.png ${figdir}/steps_radar/steps_radar_ac05A0+${i}.png -tile 2x2 -geometry +0+0 -background none tmp/1_${i}.png &
done

wait
convert -delay 100 tmp/*.png -loop 0 control_ens_mean_obs.gif


