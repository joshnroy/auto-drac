import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import sys

class ConvDrAC():
    """
    Data-regularized Actor-Critic (DrAC) object
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_id=None,
                 aug_func=None,
                 aug_coef=0.1,
                 env_name=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.actor_critic_parameters = list(actor_critic.encoder.parameters()) + list(actor_critic.actor.parameters()) + list(actor_critic.critic.parameters())
        self.model_parameters = list(actor_critic.transition_model.parameters()) + list(actor_critic.reward_model.parameters()) + list(actor_critic.encoder.parameters())

        self.optimizer_critic = optim.Adam(self.actor_critic_parameters, lr=lr, eps=eps)
        self.optimizer_model = optim.Adam(self.model_parameters, lr=lr, eps=eps)
        
        self.aug_id = aug_id
        self.aug_func = aug_func
        self.aug_coef = aug_coef
        self.model_coef = 0.

        self.env_name = env_name

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        transition_model_loss_epoch = 0
        reward_model_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, next_obs_batch = sample

# MODEL OPTIMIZATION
                with torch.no_grad():
                    next_obs_features, _ = self.actor_critic.encoder(next_obs_batch, recurrent_hidden_states_batch, masks_batch)
                    next_obs_features = torch.reshape(next_obs_features, (-1, 1) + self.actor_critic.state_shape)
                
                predicted_next_states, predicted_rewards = self.actor_critic.predict_next_state_reward(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                next_state_loss = F.mse_loss(predicted_next_states, next_obs_features)
                reward_loss = F.mse_loss(predicted_rewards, return_batch)
                model_loss = next_state_loss + reward_loss

                self.optimizer_model.zero_grad()
                (model_loss * self.model_coef).backward()
                self.optimizer_model.step()  

# -------------------

# ACTION AND VALUE OPTIMIZATION

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                                
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # self.optimizer_actor.zero_grad()
                # (action_loss - dist_entropy * self.entropy_coef).backward(retain_graph=True)
                # nn.utils.clip_grad_norm_(self.actor_parameters,
                #                         self.max_grad_norm)
                # self.optimizer_actor.step()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()

                self.optimizer_critic.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic_parameters,
                                        self.max_grad_norm)
                self.optimizer_critic.step()

# -------------------

                if False:
                    obs_batch_aug = self.aug_func.do_augmentation(obs_batch)
                    obs_batch_id = self.aug_id(obs_batch)
                    
                    _, new_actions_batch, _, _ = self.actor_critic.act(\
                        obs_batch_id, recurrent_hidden_states_batch, masks_batch)
                    values_aug, action_log_probs_aug, dist_entropy_aug, _ = \
                        self.actor_critic.evaluate_actions(obs_batch_aug, \
                        recurrent_hidden_states_batch, masks_batch, new_actions_batch)
                    # Compute Augmented Loss
                    action_loss_aug = - action_log_probs_aug.mean()
                    value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()
                else:
                    action_loss_aug = 0.
                    value_loss_aug = 0.

                # Update actor-critic using both PPO and Augmented Loss. Also update model using model loss
                # aug_loss = value_loss_aug + action_loss_aug
                # (value_loss * self.value_loss_coef + action_loss -
                #     dist_entropy * self.entropy_coef + 
                #     aug_loss * self.aug_coef + model_loss * self.model_coef).backward()
                # nn.utils.clip_grad_norm_(self.critic_parameters,
                #                         self.max_grad_norm)


                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                transition_model_loss_epoch += next_state_loss.item()
                reward_model_loss_epoch += reward_loss.item()

                # if self.aug_func:
                #     self.aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        transition_model_loss_epoch /= num_updates
        reward_model_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transition_model_loss_epoch, reward_model_loss_epoch
