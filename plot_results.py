#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm
import matplotlib
from pathlib import Path
import sys
matplotlib.use("Agg")

sns.set_style("darkgrid")

prefix = "finetuning/"

SMOOTHING_WINDOW = 20

env_names = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", 
             "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
num_wide = int(np.sqrt(len(env_names)))
fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
    original_file_naming_scheme="reconstruction/ppo_modelclip/" + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
    big_df = pd.DataFrame()
    ax = axes[i_env // num_wide, i_env % num_wide]

    things = [("pretraining/vanilla_ppo_switch1_longer/" + env_name + "/", "Pretraining")]
#     ax = axes
    for file_name in glob(original_file_naming_scheme):
        try:
            df = pd.read_csv(file_name)
            df = df[["train/mean_episode_reward", "test/mean_episode_reward", "train/total_num_steps"]]
            df.columns = ["PPO Train", "PPO Test", "Timestep"]
            df.loc[:, df.columns != "Timestep"] = df.loc[:, df.columns != "Timestep"].rolling(window=SMOOTHING_WINDOW).mean()
            df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Reward")
            big_df = big_df.append(df, ignore_index=True)
        except Exception as e:
            print(file_name, e)
    for file_naming_scheme, num in things:
        for file_name in Path(file_naming_scheme).rglob("*.csv"):
            try:
                df = pd.read_csv(file_name)
                df = df[[prefix + "train/mean_episode_reward", prefix + "test/mean_episode_reward", prefix + "train/total_num_steps"]]
                df.columns = [str(num) + " Train", str(num) + " Test", "Timestep"]
                df.loc[:, df.columns != "Timestep"] = df.loc[:, df.columns != "Timestep"].rolling(window=SMOOTHING_WINDOW).mean()
                df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Reward")
                big_df = big_df.append(df, ignore_index=True)
            except Exception as e:
                print(file_name, e)
    if len(big_df) > 0:
        sns.lineplot(data=big_df, x="Timestep", y="Reward", hue="Type", ci='sd', ax=ax)
    ax.set_title(env_name.title())

plt.suptitle("Rewards")
plt.savefig("rewards.png", bbox_inches='tight')

# TRANSITION_CLIP = 4.
# 
# fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
# for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
#     file_naming_scheme=name + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     original_file_naming_scheme="reconstruction/ppo_modelclip/" + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     big_df = pd.DataFrame()
#     ax = axes[i_env // num_wide, i_env % num_wide]
#     things = [(prefix + "vanilla_ppo_switch1/" + env_name + "/", "Pretraining")]
#     if False:
#         for file_name in glob(original_file_naming_scheme):
#             try:
#                 df = pd.read_csv(file_name)
#                 df = df[["losses/transition_model_loss", "train/total_num_steps"]]
#                 df.columns = ["PPO Transition Model", "Timestep"]
#                 df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).rolling(window=SMOOTHING_WINDOW).mean()
#                 df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#                 big_df = big_df.append(df, ignore_index=True)
#             except Exception as e:
#                 print(file_name, e)
#     for file_naming_scheme, num in things:
#         for file_name in Path(file_naming_scheme).rglob("*.csv"):
#             try:
#                 df = pd.read_csv(file_name)
#                 df = df[["losses/transition_model_loss", "train/total_num_steps"]]
#                 df.columns = [str(num) + " Transition Model", "Timestep"]
#                 df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).rolling(window=SMOOTHING_WINDOW).mean()
#                 df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#                 big_df = big_df.append(df, ignore_index=True)
#             except Exception as e:
#                 print(file_name, e)
#     if len(big_df) > 0:
#         sns.lineplot(data=big_df, x="Timestep", y="Value", hue="Type", ci='sd', ax=ax)
#     ax.set_title(env_name.title())
# 
# plt.suptitle("Transition Model Losses")
# plt.savefig("transition_model_losses.png", bbox_inches='tight')
# plt.close()
# 
# REWARD_CLIP = 10.
# 
# fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
# for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
#     file_naming_scheme=name + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     original_file_naming_scheme="reconstruction/ppo_modelclip/" + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     big_df = pd.DataFrame()
#     ax = axes[i_env // num_wide, i_env % num_wide]
#     things = [(prefix + "vanilla_ppo_switch1/" + env_name + "/", "Pretraining")]
#     if False:
#         for file_name in glob(original_file_naming_scheme):
#             try:
#                 df = pd.read_csv(file_name)
#                 df = df[["losses/reward_model_loss", "train/total_num_steps"]]
#                 df.columns = ["PPO Reward Model", "Timestep"]
#                 df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).clip(0., REWARD_CLIP).rolling(window=SMOOTHING_WINDOW).mean()
#                 df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#                 big_df = big_df.append(df, ignore_index=True)
#             except Exception as e:
#                 print(file_name, e)
#     for file_naming_scheme, num in things:
#         for file_name in Path(file_naming_scheme).rglob("*.csv"):
#             try:
#                 df = pd.read_csv(file_name)
#                 df = df[["losses/reward_model_loss", "test/reward_model_error", "train/total_num_steps"]]
#                 df.columns = [str(num) + " Train", str(num) + " Test", "Timestep"]
#                 df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).clip(0., REWARD_CLIP).rolling(window=SMOOTHING_WINDOW).mean()
#                 df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#                 big_df = big_df.append(df, ignore_index=True)
#             except Exception as e:
#                 print(file_name, e)
#     if len(big_df) > 0:
#         sns.lineplot(data=big_df, x="Timestep", y="Value", hue="Type", ci='sd', ax=ax)
#     ax.set_title(env_name.title())
# 
# plt.suptitle("Reward Model Losses")
# plt.savefig("reward_model_losses.png", bbox_inches='tight')
# plt.close()
# 
# fig, axes = plt.subplots(nrows=num_wide, ncols=num_wide, figsize=(20, 20))
# for i_env, env_name in tqdm(enumerate(env_names), total=len(env_names)):
#     file_naming_scheme=name + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     original_file_naming_scheme="reconstruction/ppo_modelclip/" + env_name + "/" + env_name + "-*/progress-drac-" + env_name + "-reproduce-s1.csv"
#     big_df = pd.DataFrame()
#     ax = axes[i_env // num_wide, i_env % num_wide]
#     things = [(prefix + "vanilla_ppo_switch1/" + env_name + "/", "Pretraining")]
#     for file_name in glob(original_file_naming_scheme):
#         try:
#             df = pd.read_csv(file_name)
#             df = df[["debug/next_obs_variance", "train/total_num_steps"]]
#             df.columns = ["PPO Next State Variance", "Timestep"]
#             df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).rolling(window=SMOOTHING_WINDOW).mean()
#             df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#             big_df = big_df.append(df, ignore_index=True)
#         except Exception as e:
#             print(file_name, e)
#     for file_naming_scheme, num in things:
#         for file_name in Path(file_naming_scheme).rglob("*.csv"):
#             try:
#                 df = pd.read_csv(file_name)
#                 df = df[["debug/next_obs_variance", "train/total_num_steps"]]
#                 df.columns = [str(num) + " Next State Variance", "Timestep"]
#                 df.loc[:, df.columns != "Timestep"] = np.sqrt(df.loc[:, df.columns != "Timestep"]).rolling(window=SMOOTHING_WINDOW).mean()
#                 df = pd.melt(df, id_vars=['Timestep'], var_name="Type", value_name="Value")
#                 big_df = big_df.append(df, ignore_index=True)
#             except Exception as e:
#                 print(file_name, e)
#     if len(big_df) > 0:
#         sns.lineplot(data=big_df, x="Timestep", y="Value", hue="Type", ci='sd', ax=ax)
#     ax.set_title(env_name.title())
# 
# plt.suptitle("Next State Variance")
# plt.savefig("state_var.png", bbox_inches='tight')
# plt.close()
