import numpy as np
import torch
import sys
import torch.nn.functional as F

from ucb_rl2_meta import utils

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from ucb_rl2_meta.envs import VecPyTorchProcgen, TransposeImageProcgen


def evaluate(args, actor_critic, device, num_processes=1, aug_id=None):
    actor_critic.eval()
    
    # Sample Levels From the Full Distribution 
    venv = ProcgenEnv(num_envs=num_processes, env_name=args.env_name, \
        num_levels=0, start_level=0, \
        distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []
    eval_reconstruction_errors = []
    eval_reward_model_errors = []

    rews = None
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            if aug_id:
                obs = aug_id(obs)
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

            features_and_actions, features = actor_critic.get_features(obs, eval_recurrent_hidden_states, eval_masks, action)
            reconstructions = actor_critic.reconstruct_observation(features)
            reconstruction_loss = F.l1_loss(reconstructions, obs)
            eval_reconstruction_errors.append(reconstruction_loss.item())

        obs, rews, done, infos = eval_envs.step(action)
        rews = rews.to(device)
        with torch.no_grad():
            if rews is not None:
                predicted_rewards = actor_critic.predict_reward(features_and_actions)
                reward_loss = F.l1_loss(predicted_rewards, rews)
                eval_reward_model_errors.append(reward_loss.item())
            else:
                eval_reward_model_errors.append(0)

         
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        .format(len(eval_episode_rewards), \
        np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    return eval_episode_rewards, eval_reconstruction_errors, eval_reward_model_errors
