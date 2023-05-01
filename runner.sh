for ((i=9; i<14; i++)); do
  python parallel_experiments.py -d $i -a 'G_design'
done
for ((i=8; i<14; i++)); do
  python parallel_experiments.py -d $i -a 'RAGE'
done
for ((i=8; i<14; i++)); do
  python parallel_experiments.py -d $i -a 'BOBW'
done