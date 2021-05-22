import numpy as np
from ..multiagentenv import MultiAgentEnv


class GroupMatching(MultiAgentEnv):
    def __init__(self, entity_scheme=True, n_agents=4, n_states=10, n_groups=2,
                 rand_trans=0.1, episode_limit=50, fixed_scen=False, seed=None):
        super(GroupMatching, self).__init__()
        assert entity_scheme, "This environment only supports entity scheme"
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_groups = n_groups
        self.rand_trans = rand_trans
        self.episode_limit = episode_limit
        self.fixed_scen = fixed_scen
        self.n_actions = 3  # left, stay, right
        self.seed(seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions[:self.n_agents]]
        for ia, ac in enumerate(actions):
            if self.random.uniform() < self.rand_trans:
                ac = self.random.randint(0, self.n_actions)
            if ac != 1:  # if not stay action
                curr_loc = np.where(self.agent_locs[ia])[0].item()
                self.agent_locs[ia, curr_loc] = 0
                if ac == 0:  # left
                    # negative indices will automatically circle to end
                    self.agent_locs[ia, curr_loc - 1] = 1
                elif ac == 2:  # right
                    next_loc = curr_loc + 1
                    if next_loc >= self.n_states:
                        next_loc -= self.n_states
                    self.agent_locs[ia, next_loc] = 1

        curr_matches = self._calc_group_piles()
        rew = -0.1  # time penalty
        rew += 2.5 * (curr_matches - self.prev_matches)
        self.prev_matches = curr_matches

        info = {'solved': False}
        done = False
        if curr_matches == self.n_groups:
            done = True
            info['solved'] = True

        self.t += 1
        if self.t == self.episode_limit:
            done = True
            info['episode_limit'] = True

        return rew, done, info

    def get_masks(self):
        obs_mask = np.zeros((self.n_agents, self.n_agents), dtype=np.uint8)
        entity_mask = np.zeros(self.n_agents, dtype=np.uint8)
        gt_mask = np.ones((self.n_agents, self.n_agents), dtype=np.uint8)
        for ia in range(self.n_agents):
            for grp in self.agent_groups:
                if ia in grp:
                    gt_mask[ia, grp] = 0
                    break
        return obs_mask, entity_mask, gt_mask

    def get_entities(self):
        locs = self.agent_locs.copy()
        groups = np.zeros((self.n_agents, self.n_groups), dtype=np.float32)
        for ig, grp in enumerate(self.agent_groups):
            groups[grp, ig] = 1
        agent_ids = np.eye(self.n_agents, dtype=np.float32)
        entities = np.concatenate((locs, groups, agent_ids), axis=1)
        return [entities[i] for i in range(self.n_agents)]

    def get_entity_size(self):
        return self.n_states + self.n_groups + self.n_agents

    def get_avail_actions(self):
        return [[1 for _ in range(self.n_actions)] for _ in range(self.n_agents)]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        agents = list(range(self.n_agents))
        if not self.fixed_scen:
            self.random.shuffle(agents)
            partitions = [0] + self.random.randint(0, self.n_agents, size=(self.n_groups - 1,)).tolist() + [self.n_agents]
        else:
            partitions = np.linspace(0, self.n_agents, self.n_groups + 1).round().astype(np.int).tolist()
        self.agent_groups = [agents[s:e] for s, e in zip(partitions[:-1], partitions[1:])]

        self.agent_locs = np.zeros((self.n_agents, self.n_states), dtype=np.float32)
        self.agent_locs[range(self.n_agents), self.random.randint(0, self.n_states, size=self.n_agents)] = 1

        self.prev_matches = self._calc_group_piles()

        self.t = 0
        return self.get_entities(), self.get_masks()

    def _calc_group_piles(self):
        return sum(self.agent_locs[g].sum(0).max() == len(g) for g in self.agent_groups)

    def close(self):
        return

    def seed(self, seed):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)

    def get_env_info(self, args):
        env_info = {"entity_shape": self.get_entity_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "n_entities": self.n_agents,
                    "gt_mask_avail": True,
                    "episode_limit": self.episode_limit}
        return env_info


if __name__ == '__main__':
    env = GroupMatching(entity_scheme=True, n_agents=4, n_states=10, n_groups=2,
                     rand_trans=0.1, episode_limit=50, seed=None)
    env.reset()
    done = False
