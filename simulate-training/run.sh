pkill python
git pull


world_size=$1
skiprate=$2
file=$3
output_dir=$4

for ((i=0; i < $world_size; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u $file $i $world_size $skiprate $output_dir >"$output_dir/out$i.txt")&


done