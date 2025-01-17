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
    CUDA_VISIBLE_DEVICES=$(($n)) nohup python train_dueling_em.py --env=${game}  --comment=dueling_em_imit_1e-1 --log_dir=./tflogs/${game} --imitate --num-steps=40000000 >& ./nohup/dueling_em_imit_1e-1_${game}.txt &
    n=$((($n+1)%${gpunum}))
done
