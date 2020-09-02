kill -9 $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == 2 && $3 > 0 {print $3}')

CUDA_VISIBLE_DEVICES=0 python3 ./dopamine/discrete_domains/train_mfec.py --base_dir=/home/hh/dopamine_logs/mfec_test/ --gin_files=/home/hh/dopamine/dopamine/agents/episodic_control/configs/mfec_cnn.gin --gin_bindings="atari_lib.create_atari_environment.game_name=\"Pong\"" --env=Pong


https://github.com/cirosantilli/cpp-cheat/blob/39fed9227337d414a12822345d2fa98535f34599/opencl/min.c#L1

