import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ..layers import EntityAttentionLayer, EntityPoolingLayer


class EntityAttentionFFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionFFAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
        else:
            self.attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(args.attn_embed_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ret_attn_logits=None):
        if len(inputs) == 3:
            entities, obs_mask, entity_mask = inputs
        else:
            entities, obs_mask, entity_mask, gt_mask = inputs
        if self.args.gt_obs_mask:
            obs_mask = gt_mask
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.fc1(entities))
        attn_outs = self.attn(x1, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits)
        attn_outs = F.relu(attn_outs)
        if ret_attn_logits is not None:
            x2, attn_logits = attn_outs
        else:
            x2 = attn_outs
        q = self.fc2(x2)
        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1), 0)
        # q = q.reshape(bs * self.args.n_agents, -1)
        if ret_attn_logits is not None:
            return q, attn_outs, attn_logits.reshape(bs, ts, self.args.n_agents, ne)
        return q, attn_outs


class ImagineEntityAttentionFFAgent(EntityAttentionFFAgent):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttentionFFAgent, self).__init__(*args, **kwargs)

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - agent_mask.to(th.float)).reshape(bs * ts, self.args.n_agents, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, self.args.n_agents, ne).to(th.uint8)

    def forward(self, inputs, hidden_state, imagine=False, use_gt_factors=False, use_rand_gt_factors=False):
        if not imagine:
            return super(ImagineEntityAttentionFFAgent, self).forward(inputs, hidden_state)
        if len(inputs) == 3:
            entities, obs_mask, entity_mask = inputs
        else:
            entities, obs_mask, entity_mask, gt_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        if use_gt_factors:
            withinattnmask = gt_mask
            interactattnmask = self.logical_not(withinattnmask)
        else:
            groupA = th.bernoulli(groupA_probs).to(th.uint8)
            groupB = self.logical_not(groupA)
            # mask out entities not present in env
            groupA = self.logical_or(groupA, entity_mask[:, [0]])
            groupB = self.logical_or(groupB, entity_mask[:, [0]])

            # convert entity mask to attention mask
            groupAattnmask = self.entitymask2attnmask(groupA)
            groupBattnmask = self.entitymask2attnmask(groupB)
            # create attention mask for interactions between groups
            interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                               self.logical_not(groupBattnmask))
            # get within group attention mask
            withinattnmask = self.logical_not(interactattnmask)
        if use_rand_gt_factors:
            assert not use_gt_factors, "Can only select one of use_rand_gt_factors and use_gt_factors"
            withinattnmask = self.logical_or(withinattnmask, gt_mask)
            interactattnmask = self.logical_not(withinattnmask)


        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(3, 1, 1, 1)
        obs_mask = th.cat([obs_mask, withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)
        hidden_state = hidden_state.repeat(3, 1, 1)
        q, h = super(ImagineEntityAttentionFFAgent, self).forward(inputs, hidden_state)
        if use_gt_factors or use_rand_gt_factors:
            rep_t = 1
        else:
            rep_t = ts
        return q, h, (Wattnmask_noobs.repeat(1, rep_t, 1, 1), Iattnmask_noobs.repeat(1, rep_t, 1, 1))
