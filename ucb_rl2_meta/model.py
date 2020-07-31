import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from ucb_rl2_meta.distributions import Categorical
from ucb_rl2_meta.utils import init


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Policy(nn.Module):
    """
    Actor-Critic module 
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(Policy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        if len(obs_shape) == 3:
            encoder = ResNetBase
        elif len(obs_shape) == 1:
            encoder = MLPBase

        self.encoder = encoder(obs_shape[0], **base_kwargs)
        self.actor = Categorical(self.encoder.output_size, num_actions)

        self.critic = []
        self.critic.append(init_relu_(nn.Linear(self.encoder.output_size, self.encoder.output_size)))
        self.critic.append(nn.ReLU(inplace=True))
        self.critic.append(init_relu_(nn.Linear(self.encoder.output_size, self.encoder.output_size)))
        self.critic.append(nn.ReLU(inplace=True))
        self.critic.append(init_(nn.Linear(self.encoder.output_size, 1)))
        self.critic = nn.Sequential(*self.critic)
        
    @property
    def is_recurrent(self):
        return self.encoder.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.encoder.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        actor_features, rnn_hxs = self.encoder(inputs, rnn_hxs, masks)
        value = self.critic(actor_features)
        dist = self.actor(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, detach_encoder=False):
        actor_features, _ = self.encoder(inputs, rnn_hxs, masks)
        if detach_encoder:
            actor_features.detach()
        value = self.critic(actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, compute_value=True, detach_encoder=False):
        actor_features, rnn_hxs = self.encoder(inputs, rnn_hxs, masks)
        if detach_encoder:
            actor_features.detach()
        if compute_value:
            value = self.critic(actor_features)
        dist = self.actor(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if compute_value:
            return value, action_log_probs, dist_entropy, rnn_hxs
        else:
            return action_log_probs, dist_entropy, rnn_hxs

class PlanningPolicy(Policy):
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(PlanningPolicy, self).__init__(obs_shape, num_actions, base_kwargs=base_kwargs)
        self.encoder = PlanningResNetBase(obs_shape[0], **base_kwargs)

class ModelBasedPolicy(Policy):
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(ModelBasedPolicy, self).__init__(obs_shape, num_actions, base_kwargs=base_kwargs)

        self.state_shape = (int(np.sqrt(self.encoder.output_size)), int(np.sqrt(self.encoder.output_size)))
        self.transition_model = TransitionModel(self.state_shape, num_actions)
        self.reward_model = RewardModel(self.state_shape, num_actions)
        self.reconstruction_model = ReconstructionModel(self.state_shape, obs_shape)

    def get_features(self, inputs, rnn_hxs, masks, actions):
        actor_features, _ = self.encoder(inputs, rnn_hxs, masks)
        actor_features = torch.reshape(actor_features, (-1, 1) + self.state_shape)
        broadcasted_actions_shape = list(actor_features.shape)
        actions = actions.float().unsqueeze(-1).unsqueeze(-1).expand(broadcasted_actions_shape)
        model_inputs = torch.cat([actor_features, actions], dim=1)

        return model_inputs, actor_features

    def predict_next_state(self, model_inputs):
        predicted_next_state = self.transition_model(model_inputs)

        return predicted_next_state

    def predict_reward(self, model_inputs):
        predicted_reward = self.reward_model(model_inputs)

        return predicted_reward

    def reconstruct_observation(self, states):
        observation = self.reconstruction_model(states)

        return observation

class ReconstructionModel(nn.Module):
    def __init__(self, state_shape, observation_shape, kernel_size=5):
        super(ReconstructionModel, self).__init__()
        
        self.state_shape = state_shape
        self.observation_shape = observation_shape
        hidden_size = 64

        layers = []
        layers.append(nn.ConvTranspose2d(1, out_channels=64, kernel_size=5, stride=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(64, out_channels=32, kernel_size=9, stride=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(32, 3, kernel_size=11, stride=2))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        output = self.model(inputs)
        output = output[:, :, :64, :64]
        return output

class TransitionModel(nn.Module):
    def __init__(self, state_shape, num_actions, kernel_size=5):
        super(TransitionModel, self).__init__()
        
        self.state_shape = state_shape
        self.num_actions = num_actions

        inner_kernel_size = 1
        hidden_size = 128

        conv_padding = int(np.floor(kernel_size / 2.))
        inner_conv_padding = int(np.floor(inner_kernel_size / 2.))

        layers = []
        layers.append(Conv2d_tf(2, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=inner_kernel_size, stride=1, padding=(inner_conv_padding,inner_conv_padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=inner_kernel_size, stride=1, padding=(inner_conv_padding,inner_conv_padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=inner_kernel_size, stride=1, padding=(inner_conv_padding,inner_conv_padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=inner_kernel_size, stride=1, padding=(inner_conv_padding,inner_conv_padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Conv2d_tf(hidden_size, 1, kernel_size=inner_kernel_size, stride=1, padding=(inner_conv_padding,inner_conv_padding)))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class RewardModel(nn.Module):
    def __init__(self, state_shape, num_actions, kernel_size=1):
        super(RewardModel, self).__init__()
        
        self.state_shape = state_shape
        self.num_actions = num_actions

        hidden_size = 128

        conv_padding = int(np.floor(kernel_size / 2.))

        layers = []

        self.conv_model = False
        if self.conv_model:
            layers.append(Conv2d_tf(2, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_tf(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_tf(hidden_size, 1, kernel_size=kernel_size, stride=1, padding=(conv_padding,conv_padding)))
        else:
            flattened_shape = self.state_shape[0] * self.state_shape[1] * 2
            intermediate_shape = 128
            layers.append(nn.Flatten())
            layers.append(init_relu_(nn.Linear(flattened_shape, intermediate_shape)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(init_relu_(nn.Linear(intermediate_shape, intermediate_shape)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(init_(nn.Linear(intermediate_shape, 1)))

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.model(inputs)
        if self.conv_model:
            return torch.sum(x, (2, 3))
        else:
            return x


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
        super(ResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = init_tanh_(nn.Linear(2048, hidden_size))
        # self.fc = init_(nn.Linear(2048, hidden_size))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))
        # x = self.fc(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = self.tanh(x)

        return x, rnn_hxs

class PlanningResNetBase(NNBase):
    """
    PlanningResidual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32], num_planning_steps=10):
        super(PlanningResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.num_planning_steps = num_planning_steps

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))

        self.model = []
        for _ in range(3):
            self.model.append(nn.Linear(hidden_size, hidden_size))
            self.model.append(nn.ReLU())
        self.model = nn.Sequential(*self.model)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        for _ in range(self.num_planning_steps):
            x = self.model(x)

        return self.critic_linear(x), x, rnn_hxs


class AugCNN(nn.Module):
    """
    Convolutional Neural Network used as Augmentation
    """
    def __init__(self):
        super(AugCNN, self).__init__()

        self.aug = Conv2d_tf(3, 3, kernel_size=3)

        apply_init_(self.modules())

        self.train()

    def forward(self, obs):
        return self.aug(obs)
