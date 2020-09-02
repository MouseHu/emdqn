#!/usr/bin/env bash

declare -a games=('MsPacman'
'Qbert'
'SpaceInvaders'
'Frostbite'
'Pong'
)
n=1
gpunum=8
for game in "${games[@]}"
do
    CUDA_VISIBLE_DEVICES=$(($n)) nohup python train_dueling_em.py --env=${game}  --gamma=1 --knn=11 --comment=dueling_em_baseline_fixmem --log_dir=./tflogs/${game} --baseline --num-steps=10000000 >& ./nohup/dueling_em_baseline_fixmem_${game}.txt &
    n=$((($n+1)%${gpunum}))
done
