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
#$ -t 1-160
#
# Runs at most 20 jobs at once
#$ -tc 20

envs=(bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot)
num_mini_batches=(16 32)

ID=$(($SGE_TASK_ID - 1))
num_mini_batch=${num_mini_batches[$(($ID / 80))]}
ID_INNER=$(($ID % 80))
env_name=${envs[$(($ID_INNER % 16))]}
trial=$(($ID_INNER / 16))

source ~/miniconda3/bin/activate && conda activate auto-drac && python train.py --env_name ${env_name} --log_dir modelbased_logs/${env_name}/${env_name}-${trial}-${num_mini_batch} --aug_coef 0. --num_mini_batch ${num_mini_batch}
