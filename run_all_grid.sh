#!/bin/bash
#
# Execute from cwd
#$ -cwd
#
# This is a day long job
#$ -l day
#
# Uses 1 GPU
#$ -l gpus=1
#
# Runs 80 jobs
#$ -t 1-80
#
# Runs at most 20 jobs at once
#$ -tc 20

envs=(bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot)

ID=$(($SGE_TASK_ID - 1))
env_name=${envs[$(($ID % 16))]}
trial=$(($ID / 16))

source ~/miniconda3/bin/activate && conda activate auto-drac && python train.py --env_name ${env_name} --log_dir ppo_logs/${env_name}/${env_name}-${trial} --aug_coef 0.
