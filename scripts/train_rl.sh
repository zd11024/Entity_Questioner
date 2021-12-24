work_path=$1
save_name=$2
guesser_path=$3
abot_path=$4
qbot_path=$5
data_path=$work_path/data
ckpt_path=$work_path/checkpoints

python -u $work_path/train.py \
    -numWorkers 4 \
    -useGPU \
    -numEpoch 12 \
    -batchSize 32 \
    -dataroot $data_path \
    -savePath $ckpt_path \
    -saveName $save_name \
    -guesserFrom $guesser_path \
    -startFrom $abot_path \
    -qstartFrom $qbot_path \
    -use_entity 1 \
    -qencoder entity_encoder \