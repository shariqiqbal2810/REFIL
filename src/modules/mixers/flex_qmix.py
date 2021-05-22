import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(hypernet_embed,
                                             hypernet_embed,
                                             hypernet_embed, args)
        else:
            self.attn = EntityPoolingLayer(hypernet_embed,
                                           hypernet_embed,
                                           hypernet_embed,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2), 0)
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, mode='scalar')

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh

    def forward(self, agent_qs, inputs, imagine_groups=None):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          ne, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          ne, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        b1 = self.hyper_b_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=-1)
        else:
            w1 = th.abs(w1)

        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Second layer
        if self.args.softmax_mixing_weights:
            w_final = F.softmax(self.hyper_w_final(entities, entity_mask), dim=-1)
        else:
            w_final = th.abs(self.hyper_w_final(entities, entity_mask))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(entities, entity_mask).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class LinearFlexQMixer(nn.Module):
    def __init__(self, args):
        super(LinearFlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='alt_vector')
        self.V = AttentionHyperNet(args, mode='scalar')

    def forward(self, agent_qs, inputs, imagine_groups=None, ret_ingroup_prop=False):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, self.n_agents * 2)
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=1)
        else:
            w1 = th.abs(w1)
        v = self.V(entities, entity_mask)

        q_cont = agent_qs * w1
        q_tot = q_cont.sum(dim=1) + v
        # Reshape and return
        q_tot = q_tot.view(bs, -1, 1)
        if ret_ingroup_prop:
            ingroup_w = w1.clone()
            ingroup_w[:, self.n_agents:] = 0  # zero-out out of group weights
            ingroup_prop = (ingroup_w.sum(dim=1)).mean()
            return q_tot, ingroup_prop
        return q_tot
