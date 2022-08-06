
counter=0
while [ $counter -le 100 ]
do
  counter_str=$counter
  while [ ${#counter_str} -ne 4 ];
    do
    counter_str="0"$counter_str
  done
  tmp="scene"$counter_str"_00"
  echo "Processing: "$tmp
  /home/deltamarine/COMP9491/venv37/bin/python inference.py --model results/release/semseg/final.ckpt --scenes processed_data/sample/$tmp/info.json
  counter=$((counter+1))
done