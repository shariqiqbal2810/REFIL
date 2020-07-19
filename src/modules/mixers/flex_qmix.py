import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions

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
        if args.dropout_prob > 0:
            self.drop = nn.Dropout(p=args.dropout_prob)
        else:
            self.drop = nn.Identity()
        if args.norm_layer == 'layernorm':
            self.norm_lyr = nn.LayerNorm(hypernet_embed)
        elif args.norm_layer == 'batchnorm':
            self.norm_lyr = nn.BatchNorm1d(hypernet_embed)
        else:
            self.norm_lyr = nn.Identity()
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
        drop_x2 = self.drop(x2)
        if self.args.norm_layer == 'batchnorm':
            drop_x2 = drop_x2.reshape(-1, self.args.hypernet_embed)
        norm_x2 = self.norm_lyr(drop_x2)
        if self.args.norm_layer == 'batchnorm':
            norm_x2 = norm_x2.reshape(-1, self.args.n_agents, self.args.hypernet_embed)
        x3 = self.fc2(norm_x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2), 0)
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3


class MaskedHyperNet(nn.Module):
    """
    Normal QMIX but with masked entities (to handle dynamic populations)
    """
    def __init__(self, args, mode='matrix'):
        super(MaskedHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions

        hypernet_embed = args.hypernet_embed
        if mode == 'matrix':
            out_dim = args.mixing_embed_dim * args.n_agents
        elif mode == 'vector':
            out_dim = args.mixing_embed_dim
        elif mode == 'scalar':
            out_dim = 1
        self.fc1 = nn.Linear(self.entity_dim * args.n_entities, hypernet_embed)
        self.fc2 = nn.Linear(hypernet_embed, out_dim)

    def forward(self, entities, entity_mask):
        bs, ne, ed = entities.shape
        entities = entities.masked_fill(entity_mask.unsqueeze(2), 0)
        entities = entities.reshape(bs, -1)
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        x2 = self.fc2(x1)
        if self.mode == 'matrix':
            x2 = x2.reshape(bs, self.args.n_agents, self.args.mixing_embed_dim)
            x2 = x2.masked_fill(agent_mask.unsqueeze(2), 0)
        return x2


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        if args.attention_hyper:
            hypernet_class = AttentionHyperNet
        else:
            hypernet_class = MaskedHyperNet
        self.hyper_w_1 = hypernet_class(args, mode='matrix')
        self.hyper_w_final = hypernet_class(args, mode='vector')
        self.hyper_b_1 = hypernet_class(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = hypernet_class(args, mode='scalar')

        # Initialise the hyper networks with a fixed variance, if specified
        # if self.args.hyper_initialization_nonzeros > 0:
        #     std = self.args.hyper_initialization_nonzeros ** -0.5
        #     self.hyper_w_1.weight.data.normal_(std=std)
        #     self.hyper_w_1.bias.data.normal_(std=std)
        #     self.hyper_w_final.weight.data.normal_(std=std)
        #     self.hyper_w_final.bias.data.normal_(std=std)

        if self.args.gated:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

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
                                                          self.n_agents, ne))
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          self.n_agents, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=-1)
        else:
            w1 = th.abs(w1)
        b1 = self.hyper_b_1(entities, entity_mask)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)
        # Second layer
        if self.args.softmax_mixing_weights:
            w_final = F.softmax(self.hyper_w_final(entities, entity_mask), dim=-1)
        else:
            w_final = th.abs(self.hyper_w_final(entities, entity_mask))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(entities, entity_mask).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.args.skip_connections:
            s = agent_qs.sum(dim=2, keepdim=True)

        if self.args.gated:
            y = th.bmm(hidden, w_final) * self.gate + v + s
        else:
            # Compute final output
            y = th.bmm(hidden, w_final) + v + s
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
