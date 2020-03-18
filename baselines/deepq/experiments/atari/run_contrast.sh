#!/usr/bin/env bash

declare -a games=('MsPacman'
'Qbert'
'SpaceInvaders'
'Frostbite'
'Pong'
)
n=0
gpunum=8
for game in "${games[@]}"
do
    CUDA_VISIBLE_DEVICES=$(($n)) nohup python train_contrast.py --env=${game} --end_training=0 --comment=not_learning --log_dir=./tflogs/${game} --mode=max --video_path=./videos/${game} --num-steps=50000000 >& nohup/not-learning_${game}.txt &
    n=$((($n+1)%${gpunum}))
done
