

example to run prioritized sweeping with representation learning:
```
CUDA_VISIBLE_DEVICES=6 python train_prioritized_learning.py --env=vast --env_name=pong --log_dir=./tflogs/prior_pong --comment=ps_target_model --base_log_dir=/data/hh/ecbp --num-steps=3000000 --buffer-size=100000
```
example to run episodic control baseline:
```
CUDA_VISIBLE_DEVICES=6 python train_ec.py --env=vast --env_name=pong --log_dir=./tflogs/prior_pong --comment=baseline --base_log_dir=/data/hh/ecbp --num-steps=3000000 --buffer-size=100000

```

for other options, see baselines/ecbp/utils.py

for other agents like kernel based agents, you can modifiy train_prioritized_learning.py


