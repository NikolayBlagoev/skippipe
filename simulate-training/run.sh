pkill python
git pull


world_size=$1
skiprate=$2

for ((i=0; i < $world_size; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "trainer.py" $i $world_size $skiprate >"out$i.txt")&


done