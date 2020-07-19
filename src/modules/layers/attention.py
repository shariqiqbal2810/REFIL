import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EntityAttentionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, args):
        super(EntityAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_heads = args.attn_n_heads
        self.n_agents = args.n_agents
        self.args = args

        assert self.embed_dim % self.n_heads == 0, "Embed dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer('scale_factor',
                             th.scalar_tensor(self.head_dim).sqrt())

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 3, bias=False)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None, ret_attn_logits=None):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents are not available. Zero out their outputs to
                   prevent gradients from flowing back.
            shape: batch size, # of agents
        ret_attn_logits: whether to return attention logits
            None: do not return
            "max": take max over heads
            "mean": take mean over heads

        Return shape: batch size, # of agents, embedding dimension
        """
        bs, ne, ed = entities.shape
        entities_t = entities.transpose(0, 1)
        query, key, value = self.in_trans(entities_t).chunk(3, dim=2)

        query = query[:self.n_agents]

        query_spl = query.reshape(self.n_agents, bs * self.n_heads, self.head_dim).transpose(0, 1)
        key_spl = key.reshape(ne, bs * self.n_heads, self.head_dim).permute(1, 2, 0)
        value_spl = value.reshape(ne, bs * self.n_heads, self.head_dim).transpose(0, 1)

        attn_logits = th.bmm(query_spl, key_spl) / self.scale_factor
        if pre_mask is not None:
            pre_mask_rep = pre_mask.repeat_interleave(self.n_heads, dim=0)
            masked_attn_logits = attn_logits.masked_fill(pre_mask_rep, -float('Inf'))
        attn_weights = F.softmax(masked_attn_logits, dim=2)
        # some weights might be NaN (if agent is inactive and all entities were masked)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)
        attn_outs = th.bmm(attn_weights, value_spl)
        attn_outs = attn_outs.transpose(
            0, 1).reshape(self.n_agents, bs, self.embed_dim).transpose(0, 1)
        attn_outs = self.out_trans(attn_outs)
        if post_mask is not None:
            attn_outs = attn_outs.masked_fill(post_mask.unsqueeze(2), 0)
        if ret_attn_logits is not None:
            # bs * n_heads, na, ne
            attn_logits = attn_logits.reshape(bs, self.n_heads,
                                              self.n_agents, ne)
            if ret_attn_logits == 'max':
                attn_logits = attn_logits.max(dim=1)[0]
            elif ret_attn_logits == 'mean':
                attn_logits = attn_logits.mean(dim=1)
            elif ret_attn_logits == 'norm':
                attn_logits = attn_logits.mean(dim=1)
            return attn_outs, attn_logits
        return attn_outs


class EntityPoolingLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, pooling_type, args):
        super(EntityPoolingLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.pooling_type = pooling_type
        self.n_agents = args.n_agents
        self.args = args

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None, ret_attn_logits=None):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before pooling.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents are not available. Zero out their outputs to
                   prevent gradients from flowing back.
            shape: batch size, # of agents
        ret_attn_logits: not used, here to match attention layer args

        Return shape: batch size, # of agents, embedding dimension
        """
        bs, ne, ed = entities.shape

        ents_trans = self.in_trans(entities)
        
        # duplicate all entities per agent so we can mask separately
        ents_trans_rep = ents_trans.reshape(bs, 1, ne, ed).repeat(1, self.n_agents, 1, 1)

        if pre_mask is not None:
            ents_trans_rep = ents_trans_rep.masked_fill(pre_mask.unsqueeze(3), 0)

        if self.pooling_type == 'max':
            pool_outs = ents_trans_rep.max(dim=2)[0]
        elif self.pooling_type == 'mean':
            pool_outs = ents_trans_rep.mean(dim=2)

        pool_outs = self.out_trans(pool_outs)

        if post_mask is not None:
            pool_outs = pool_outs.masked_fill(post_mask.unsqueeze(2), 0)

        if ret_attn_logits is not None:
            return pool_outs, None
        return pool_outs
