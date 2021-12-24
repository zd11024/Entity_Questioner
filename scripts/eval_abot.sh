work_path=$1
abot_path=$2

python -u $work_path/evaluate.py -useGPU -useNDCG \
    -evalMode ABotRank \
    -beamSize 5 \
    -dataroot $work_path/data \
    -startFrom $abot_path \