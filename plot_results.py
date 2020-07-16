#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

sns.set_style("darkgrid")

name = "ppo_bigvanilla_two_optimizers_logs/"

env_names = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", 
             "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
num_wide = int(np.sqrt(len(env_names)))
fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
    file_naming_scheme=name + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
    original_file_naming_scheme="original_ppo_logs/" + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
    big_df = pd.DataFrame()
    ax = axes[i_env // num_wide, i_env % num_wide]
#     ax = axes
    for file_name in glob(original_file_naming_scheme):
        try:
            df = pd.read_csv(file_name)
            df = df[["train/mean_episode_reward", "test/mean_episode_reward", "train/total_num_steps"]]
            df.columns = ["PPO Train", "PPO Test", "Timestep"]
            df.loc[:, df.columns != "Timestep"] = df.loc[:, df.columns != "Timestep"].rolling(window=20).mean()
            df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Reward")
            big_df = big_df.append(df, ignore_index=True)
        except Exception as e:
            print(file_name, e)
    for file_name in glob(file_naming_scheme):
        try:
            df = pd.read_csv(file_name)
            df = df[["train/mean_episode_reward", "test/mean_episode_reward", "train/total_num_steps"]]
            df.columns = ["Our Train", "Our Test", "Timestep"]
            df.loc[:, df.columns != "Timestep"] = df.loc[:, df.columns != "Timestep"].rolling(window=20).mean()
            df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Reward")
            big_df = big_df.append(df, ignore_index=True)
        except Exception as e:
            print(file_name, e)
    if len(big_df) > 0:
        sns.lineplot(data=big_df, x="Timestep", y="Reward", hue="Type", ci='sd', ax=ax)
    ax.set_title(env_name.title())

plt.savefig("rewards.png")

REWARD_CLIP = 10.

fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
    file_naming_scheme=name + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
    big_df = pd.DataFrame()
    ax = axes[i_env // num_wide, i_env % num_wide]
#     ax = axes
    for file_name in glob(original_file_naming_scheme):
        try:
            df = pd.read_csv(file_name)
            df = df[["losses/transition_model_loss", "losses/reward_model_loss", "train/total_num_steps"]]
            df.columns = ["PPO Transition Model", "PPO Reward Model", "Timestep"]
            df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).clip(0., REWARD_CLIP).rolling(window=1).mean()
            df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
            big_df = big_df.append(df, ignore_index=True)
        except Exception as e:
            print(file_name, e)
    for file_name in glob(file_naming_scheme):
        try:
            df = pd.read_csv(file_name)
            df = df[["losses/transition_model_loss", "losses/reward_model_loss", "train/total_num_steps"]]
            df.columns = ["Our Transition Model", "Our Reward Model", "Timestep"]
            df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).clip(0., REWARD_CLIP).rolling(window=1).mean()
            df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
            big_df = big_df.append(df, ignore_index=True)
        except Exception as e:
            print(file_name, e)
    if len(big_df) > 0:
        sns.lineplot(data=big_df, x="Timestep", y="Value", hue="Type", ci='sd', ax=ax)
    ax.set_title(env_name.title())

plt.savefig("model_losses.png")
plt.close()
