from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class EntityMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAC, self).__init__(scheme, groups, args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entities = []
        entities.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entities.append(ent_acs)
        entities = th.cat(entities, dim=3)
        return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t])

    def _get_input_shape(self, scheme):
        input_shape = scheme["entities"]["vshape"]
        if self.args.entity_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape
