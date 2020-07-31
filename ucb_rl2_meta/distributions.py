import math

import torch
import torch.nn as nn

from ucb_rl2_meta.utils import init

class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu'))

        self.linear = []
        self.linear.append(init_relu_(nn.Linear(num_inputs, num_inputs)))
        self.linear.append(nn.ReLU(inplace=True))
        self.linear.append(init_relu_(nn.Linear(num_inputs, num_inputs)))
        self.linear.append(nn.ReLU(inplace=True))
        self.linear.append(init_(nn.Linear(num_inputs, num_outputs)))

        self.linear = nn.Sequential(*self.linear)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
