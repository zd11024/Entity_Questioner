work_path=$1
guesser_path=$2
abot_path=$3
qbot_path=$4

python -u $work_path/evaluate.py -useGPU -useNDCG \
    -evalMode QABotsRank \
    -beamSize 5 \
    -dataroot $work_path/data \
    -guesserFrom $guesser_path \
    -startFrom $abot_path \
    -qstartFrom $qbot_path \
    -use_entity 1 \
    -qencoder entity_encoder
