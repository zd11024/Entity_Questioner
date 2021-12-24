work_path=$1
guesser_path=$2

python -u $work_path/evaluate.py -useGPU -useNDCG \
    -evalMode GuesserRank \
    -beamSize 5 \
    -dataroot $work_path/data \
    -guesserFrom $guesser_path \