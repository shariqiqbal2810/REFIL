import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        if getattr(self.args, "hypernet_layers", 1) > 1:
            assert self.args.hypernet_layers == 2, "Only 1 or 2 hypernet_layers is supported atm!"
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

        if hasattr(self.args, 'state_masks'):
            self.register_buffer('state_masks', th.tensor(self.args.state_masks))

    def forward(self, agent_qs, states, imagine_groups=None):
        bs, max_t, sd = states.shape

        states = states.reshape(-1, self.state_dim)
        if imagine_groups is not None:
            ne = self.state_masks.shape[0]
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2)
            groupA, groupB = imagine_groups
            groupA = groupA.reshape(bs * max_t, ne, 1)
            groupB = groupB.reshape(bs * max_t, ne, 1)
            groupA_mask = (groupA * self.state_masks.reshape(1, ne, sd)).sum(dim=1).clamp_max(1)
            groupB_mask = (groupB * self.state_masks.reshape(1, ne, sd)).sum(dim=1).clamp_max(1)
            groupA_states = states * groupA_mask
            groupB_states = states * groupB_mask

            w1_A = self.hyper_w_1(groupA_states)
            w1_B = self.hyper_w_1(groupB_states)
            w1 = th.cat([w1_A, w1_B], dim=1)
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            w1 = self.hyper_w_1(states)
        # First layer
        b1 = self.hyper_b_1(states)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=-1)
        else:
            w1 = th.abs(w1)

        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Second layer
        if self.args.softmax_mixing_weights:
            w_final = F.softmax(self.hyper_w_final(states), dim=-1)
        else:
            w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
