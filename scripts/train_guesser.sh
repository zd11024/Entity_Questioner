work_path=$1
save_name=$2
data_path=$work_path/data
ckpt_path=$work_path/checkpoints

python -u $work_path/train_guesser.py \
    -numWorkers 4 \
    -useGPU \
    -numEpoch 10 \
    -featLossCoeff 1 \
    -dropout 0 \
    -use_candidate_image 1 \
    -negative_samples batch,similar \
    -dataroot $data_path \
    -savePath $ckpt_path \
    -saveName $save_name