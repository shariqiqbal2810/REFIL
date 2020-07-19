from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from ..multiagentenv import MultiAgentEnv


class Render(object):
    def __init__(self, figsize=(15, 15), dpi=48):
        self.figsize = figsize
        self.dpi = dpi
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.artists = []

    def add_artist(self, artist):
        self.artists.append(artist)

    def new_frame(self):
        self.fig.clear()
        self.ax = self.fig.gca()
        self.ax.clear()
        for artist in self.artists:
            artist.remove()
        self.artists = []
        self.ax.set_xlim(-0.1, MAP_SIZE + 1.1)
        self.ax.set_ylim(-0.1, MAP_SIZE + 1.1)
        self.ax.axis('off')

    def draw(self):
        for artist in self.artists:
            self.ax.add_artist(artist)
        self.canvas.draw()       # draw the canvas, cache the renderer
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image

    # def render(self, img):
    #     plt.figure(figsize=self.figsize)
    #     ax = plt.gca()
    #     ax.axis('off')
    #     plt.imshow(img)


class Entity(object):
    def __init__(self, obj_id, ent_id, ent_type):
        self.obj_id = obj_id
        self.ent_id = ent_id
        self.ent_type = ent_type
        self.x = None
        self.y = None


class Building(Entity):
    def __init__(self, x, y, obj_id, ent_id, game_config, burn_rate):
        super(Building, self).__init__(obj_id, ent_id, ent_type='building')
        self.x = x
        self.y = y
        self.burn_rate = burn_rate
        self.game_config = game_config
        self.health = 0.25 + (np.random.rand() * 0.5)
        self.fire_strength = 0.0
        self.complete = (self.health == 1.0)
        self.burned_down = (self.health == 0.0)

        self.fire_decrement = 0.0
        self.build_delta = 0.0

        self.fire_start_rate = self.game_config['fire_start_rate']
        self.time_to_next_fire = int(np.random.exponential(self.fire_start_rate))

    def reduce_fire(self, fire_speed):
        self.fire_decrement += fire_speed * self.game_config['fire_reduce_rate']

    def build(self, build_speed):
        self.build_delta += build_speed * self.game_config['base_build_rate']

    def set_fire(self):
        self.fire_strength = 1.0

    def step(self):
        info = {'fires_extinguished': 0,
                'buildings_completed': 0,
                'buildings_burned': 0,
                'buildings_health_delta': 0}
        if self.fire_decrement == 0.0:
            new_fire = self.game_config['fire_growth_rate'] * self.fire_strength
        else:
            new_fire = self.fire_strength - self.fire_decrement
        prev_fire = self.fire_strength
        self.fire_strength = min(1.0, max(0.0, new_fire))
        self.build_delta -= self.fire_strength * self.burn_rate * self.game_config['fire_burn_rate']
        new_health = self.health + self.build_delta

        prev_health = self.health
        self.health = min(1.0, max(0.0, new_health))
        true_delta = self.health - prev_health

        if prev_fire > 0.0 and self.fire_strength == 0.0:
            info['fires_extinguished'] += 1

        if prev_health < 1.0 and self.health == 1.0:
            info['buildings_completed'] += 1

        if prev_health > 0.0 and self.health == 0.0:
            info['buildings_burned'] += 1
            self.fire_strength = 0.0

        info['buildings_health_delta'] += true_delta

        self.fire_decrement = 0.0
        self.build_delta = 0.0

        self.complete = (self.health == 1.0)
        self.burned_down = (self.health == 0.0)

        if not (self.complete or self.burned_down):
            self.time_to_next_fire -= 1
            if self.time_to_next_fire <= 0:
                self.set_fire()
                self.time_to_next_fire = int(np.random.exponential(self.fire_start_rate))
        else:
            self.time_to_next_fire = 1000
        return info


class FastBurnBuilding(Building):
    def __init__(self, x, y, obj_id, game_config):
        super(FastBurnBuilding, self).__init__(x=x, y=y,
                                               obj_id=obj_id,
                                               ent_id=0,
                                               game_config=game_config,
                                               burn_rate=1.0)


class SlowBurnBuilding(Building):
    def __init__(self, x, y, obj_id, game_config):
        super(SlowBurnBuilding, self).__init__(x=x, y=y,
                                               obj_id=obj_id,
                                               ent_id=1,
                                               game_config=game_config,
                                               burn_rate=0.25)


class Agent(Entity):
    def __init__(self, obj_id, ent_id, fire_speed, build_speed, max_move_dist,
                 sight_range):
        super(Agent, self).__init__(obj_id, ent_id, ent_type='agent')
        self.fire_speed = fire_speed
        self.build_speed = build_speed
        self.max_move_dist = max_move_dist
        self.sight_range = sight_range
        self.prev_x = None
        self.prev_y = None
        self.build_actions = 0
        self.fire_actions = 0
        self.last_action = None


class FireFighter(Agent):
    def __init__(self, obj_id):
        super(FireFighter, self).__init__(obj_id=obj_id,
                                          ent_id=2,
                                          fire_speed=1.0,
                                          build_speed=0.05,
                                          max_move_dist=1,
                                          sight_range=None)


class Builder(Agent):
    def __init__(self, obj_id):
        super(Builder, self).__init__(obj_id=obj_id,
                                      ent_id=3,
                                      fire_speed=0.05,
                                      build_speed=1.0,
                                      max_move_dist=1,
                                      sight_range=None)


class Generalist(Agent):
    def __init__(self, obj_id):
        super(Generalist, self).__init__(obj_id=obj_id,
                                         ent_id=4,
                                         fire_speed=0.25,
                                         build_speed=0.25,
                                         max_move_dist=2,
                                         sight_range=None)


AGENT_TYPES = {'F': FireFighter, 'B': Builder, 'G': Generalist}

MAP_SIZE = 16

BUILDING_TYPES = {'F': FastBurnBuilding, 'S': SlowBurnBuilding}
N_BUILDING_TYPES = 2


class FireFightersEnv(MultiAgentEnv):
    def __init__(self,
                 entity_scheme=True,
                 scenario_dict=None,
                 episode_limit=150,
                 game_config=None,
                 reward_config=None,
                 reward_scale=20,
                 track_ac_type=False,
                 seed=0):
        if scenario_dict == 'infinite':
            self.train_scenarios = None
            self.test_scenarios = None
            self.bld_spacing = 7
            self.max_n_agents = 8
            self.max_n_buildings = 8
        else:
            self.train_scenarios = scenario_dict['train_scenarios']
            self.test_scenarios = scenario_dict['test_scenarios']
            self.max_n_agents = scenario_dict['max_n_agents']
            self.max_n_buildings = scenario_dict['max_n_buildings']
            self.bld_spacing = scenario_dict['bld_spacing']

        self.bld_pos_candidates = list(product(range(1, MAP_SIZE, self.bld_spacing),
                                               range(1, MAP_SIZE, self.bld_spacing)))

        self.episode_limit = episode_limit
        if game_config is None:
            self.game_config = {'fire_start_rate': 40,  # rate of fires starting
                                'fire_burn_rate': 0.05,  # rate at which fire burns building
                                'fire_reduce_rate': 0.25,  # rate at which agents reduce fires
                                'fire_growth_rate': 1.1,  # rate at which fire grows on its own
                                'base_build_rate': 0.25,  # rate at which agents can build
                                'fire_warn_time': 5}  # number of steps prior to fire starting that agents can detect it
        else:
            self.game_config = game_config
        if reward_config is None:
            self.reward_config = {'health_delta_mult': 0.1, 'complete': 1.0,
                                  'burned_down': -1.0, 'extinguish': 0.5,
                                  'solved': 20.0}
        else:
            self.reward_config = reward_config
        self.max_reward = 0.5 * self.reward_config['health_delta_mult'] * self.max_n_buildings
        self.max_reward += self.reward_config['complete'] * self.max_n_buildings
        self.max_reward += self.reward_config['solved']

        self.reward_scale = reward_scale
        self.track_ac_type = track_ac_type

        max_move_dist = max(a(0).max_move_dist for a in AGENT_TYPES.values())
        self.n_actions = 1 + (4 * max_move_dist) + 2  # stay, move NESW (up to max_move_dist), put out fire, build

        self._render = None
        self.time = 0
        self.seed(seed)
        self.grid = np.array([[None for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)])
        self.agents = []
        self.buildings = []

    @property
    def objects(self):
        return self.agents + self.buildings

    def _vis_objs(self, x, y, vis_range):
        gridcopy = self.grid.copy()
        min_x = max(0, x - vis_range)
        max_x = min(MAP_SIZE, x + vis_range + 1)
        min_y = max(0, y - vis_range)
        max_y = min(MAP_SIZE, y + vis_range + 1)
        local_region = gridcopy[min_x:max_x, min_y:max_y].flatten().tolist()
        return [obj for obj in local_region if obj is not None]

    def _get_dir_obj(self, x, y, direction, dist=1):
        if direction == 'N':
            if y + dist >= MAP_SIZE:
                return 'boundary'
            else:
                return self.grid[x, y + dist]
        elif direction == 'E':
            if x + dist >= MAP_SIZE:
                return 'boundary'
            else:
                return self.grid[x + dist, y]
        elif direction == 'S':
            if y - dist < 0:
                return 'boundary'
            else:
                return self.grid[x, y - dist]
        elif direction == 'W':
            if x - dist < 0:
                return 'boundary'
            else:
                return self.grid[x - dist, y]

    def _update_grid_pos(self):
        self.grid[:, :] = None
        for agent in self.agents:
            self.grid[agent.x, agent.y] = agent
        for building in self.buildings:
            self.grid[building.x, building.y] = building

    def seed(self, seed=None):
        np.random.seed(seed)

    def sample(self, test=False):
        if self.train_scenarios is None:
            n_agents = np.random.randint(2, self.max_n_agents + 1)
            n_buildings = np.random.randint(2, self.max_n_buildings + 1)
            agent_units = [np.random.choice(list(AGENT_TYPES.keys())) for _ in range(n_agents)]
            building_units = [np.random.choice(list(BUILDING_TYPES.keys())) for _ in range(n_buildings)]
            self.curr_scenario = (agent_units, building_units)
            return
        if test:
            ind = np.random.randint(len(self.test_scenarios))
            self.curr_scenario = self.test_scenarios[ind]
        else:
            ind = np.random.randint(len(self.train_scenarios))
            self.curr_scenario = self.train_scenarios[ind]

    def set_index(self, index, test=False):
        if test:
            self.curr_scenario = self.test_scenarios[index]
        else:
            self.curr_scenario = self.train_scenarios[index]

    def reset(self, index=None, test=False):
        if index is None:
            self.sample(test=test)
        else:
            self.set_index(index, test=test)

        self.agent_units, building_spec = self.curr_scenario
        self.n_agents = len(self.agent_units)
        if len(building_spec) == 2 and type(building_spec[0]) is int:
            min_n_blds, max_n_blds = building_spec
            self.n_buildings = np.random.randint(min_n_blds, max_n_blds + 1)
            self.building_units = [np.random.choice(sorted(BUILDING_TYPES.keys()))
                                   for _ in range(self.n_buildings)]
        else:
            self.n_buildings = len(building_spec)
            self.building_units = building_spec

        self.agents = []
        self.buildings = []
        self.ep_info = {'fires_extinguished': 0,
                        'buildings_completed': 0,
                        'buildings_burned': 0,
                        'buildings_health_delta': 0}

        self.time = 0

        agt_pos_grid = np.array([[None for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)])

        obj_id = 0

        # prevent agents from spawning where a building might exist
        bx, by = list(zip(*self.bld_pos_candidates))
        agt_pos_grid[bx, by] = ' '
        # shrink agent pos candidates as small as possible so agents start together
        shrink_grid = agt_pos_grid.copy()
        idx = 0
        while (shrink_grid == None).sum() >= self.n_agents:
            agt_pos_grid = shrink_grid.copy()
            shrink_grid[idx, :] = ' '
            shrink_grid[-idx - 1, :] = ' '
            shrink_grid[:, idx] = ' '
            shrink_grid[:, -idx - 1] = ' '
            idx += 1

        agt_pos_candidates = list(zip(*np.where(agt_pos_grid == None)))

        for au in self.agent_units:
            agent = AGENT_TYPES[au](obj_id=obj_id)
            agent.x, agent.y = agt_pos_candidates.pop(
                np.random.randint(len(agt_pos_candidates)))
            self.agents.append(agent)
            obj_id += 1

        bld_pos_candidates_rem = self.bld_pos_candidates.copy()
        for bu in self.building_units:
            x, y = bld_pos_candidates_rem.pop(
                np.random.randint(len(bld_pos_candidates_rem)))
            building = BUILDING_TYPES[bu](x=x, y=y,
                                          obj_id=obj_id,
                                          game_config=self.game_config)
            self.buildings.append(building)
            obj_id += 1

        self._update_grid_pos()

        return self.get_entities(), self.get_masks()

    def get_entities(self):
        nf_entity = self.get_entity_size()
        all_entities = []
        avail_actions = self.get_avail_actions()
        for obj in self.objects:
            ind = 0
            # entity type
            curr_ent = np.zeros(nf_entity, dtype=np.float32)
            curr_ent[ind + obj.ent_id] = 1
            ind += N_BUILDING_TYPES + len(AGENT_TYPES)
            # x-y loc
            curr_ent[ind + obj.x] = 1
            ind += MAP_SIZE
            curr_ent[ind + obj.y] = 1
            ind += MAP_SIZE
            # avail actions
            if obj.ent_type == 'agent':
                for iac in range(self.n_actions):
                    curr_ent[ind + iac] = avail_actions[obj.obj_id][iac]
            ind += self.n_actions
            # building properties
            if obj.ent_type == 'building':
                curr_ent[ind] = obj.health
                curr_ent[ind + 1] = int(obj.complete)
                curr_ent[ind + 2] = int(obj.burned_down)
                curr_ent[ind + 3] = obj.fire_strength
                curr_ent[ind + 4] = int(obj.fire_strength > 0.0)
                if obj.time_to_next_fire <= self.game_config['fire_warn_time']:
                    curr_ent[ind + 5] = (obj.time_to_next_fire /
                                         self.game_config['fire_warn_time'])
            all_entities.append(curr_ent)
            # pad entities to fixed number across episodes (for easier batch processing)
            if obj.obj_id == self.n_agents - 1:
                all_entities += [np.zeros(nf_entity, dtype=np.float32)
                                 for _ in range(self.max_n_agents -
                                                self.n_agents)]
            elif obj.obj_id == self.n_agents + self.n_buildings - 1:
                all_entities += [np.zeros(nf_entity, dtype=np.float32)
                                 for _ in range(self.max_n_buildings -
                                                self.n_buildings)]
        return all_entities

    def get_entity_size(self):
        nf_entity = 0
        # entity type
        nf_entity += len(AGENT_TYPES) + N_BUILDING_TYPES
        # one-hot location coordinates
        nf_entity += 2 * MAP_SIZE
        # available actions (only for agents)
        nf_entity += self.n_actions
        # building only properties (build completion, fire level)
        nf_entity += 6
        return nf_entity

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                avail_agent = self.get_avail_agent_actions(agent_id)
            else:
                avail_agent = [1] + [0] * (self.n_actions - 1)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, a_id):
        agent = self.agents[a_id]
        # stay, move NESW, put out fire, build
        avail_actions = [0] * self.n_actions
        avail_actions[0] = 1

        for dist in range(1, agent.max_move_dist + 1):
            n_obj = self._get_dir_obj(agent.x, agent.y, 'N', dist=dist)
            e_obj = self._get_dir_obj(agent.x, agent.y, 'E', dist=dist)
            s_obj = self._get_dir_obj(agent.x, agent.y, 'S', dist=dist)
            w_obj = self._get_dir_obj(agent.x, agent.y, 'W', dist=dist)
            if dist == 1:
                surr_objs = [n_obj, e_obj, s_obj, w_obj]
            # can move into space if no building/agent occupies
            ind_start = 1 + (dist - 1) * 4
            avail_actions[ind_start] = int(n_obj is None)
            avail_actions[ind_start + 1] = int(e_obj is None)
            avail_actions[ind_start + 2] = int(s_obj is None)
            avail_actions[ind_start + 3] = int(w_obj is None)
        for obj in surr_objs:
            if obj is not None and obj != 'boundary' and obj.ent_type == 'building':
                # can only put out fire if fire exists
                if obj.fire_strength > 0.0:
                    avail_actions[-2] = 1
                # can only build if building is not complete/burned down and not on fire
                if not (obj.burned_down or obj.complete) and obj.fire_strength == 0.0:
                    avail_actions[-1] = 1
                # only max of one building surrounding due to positioning constraints
                break
        return avail_actions

    def get_masks(self):
        """
        Returns:
        1) per agent observability mask over all entities (unoberserved = 1, else 0)
        3) mask of inactive entities (including enemies) over all possible entities
        """
        obs_mask = np.ones((self.n_agents, self.n_agents + self.n_buildings),
                           dtype=np.uint8)
        for agent in self.agents:
            if agent.sight_range is not None:
                vis_objs = self._vis_objs(agent.x, agent.y, agent.sight_range)
            else:
                vis_objs = self.objects
            for obj in vis_objs:
                obs_mask[agent.obj_id, obj.obj_id] = 0

        obs_mask_padded = np.ones((self.max_n_agents,
                                   self.max_n_agents + self.max_n_buildings),
                                  dtype=np.uint8)
        obs_mask_padded[:self.n_agents,
                        :self.n_agents] = obs_mask[:, :self.n_agents]
        obs_mask_padded[:self.n_agents,
                        self.max_n_agents:self.max_n_agents + self.n_buildings] = (
                            obs_mask[:, self.n_agents:]
        )
        entity_mask = np.ones(self.max_n_agents + self.max_n_buildings,
                              dtype=np.uint8)
        entity_mask[:self.n_agents] = 0
        entity_mask[self.max_n_agents:self.max_n_agents + self.n_buildings] = 0
        return obs_mask_padded, entity_mask

    def step(self, actions):
        actions = [int(a) for a in actions[:self.n_agents]]
        for agent in self.agents:
            avail_actions = self.get_avail_agent_actions(agent.obj_id)
            assert avail_actions[actions[agent.obj_id]] == 1, \
                "Agent {} cannot perform action {}".format(agent.obj_id,
                                                           actions[agent.obj_id])

        total_reward = 0
        # try initial moving actions
        for agent in self.agents:
            action = actions[agent.obj_id]
            agent.prev_x = agent.x
            agent.prev_y = agent.y
            # stay, move NESW, put out fire, build
            # movement actions
            if action == 0:
                agent.last_action = ' '
            for move_dist in range(1, agent.max_move_dist + 1):
                # can move into space if no building/agent occupies
                ind_start = 1 + (move_dist - 1) * 4
                if action == ind_start:
                    agent.y += move_dist
                    agent.last_action = 'N%i' % (move_dist)
                elif action == ind_start + 1:
                    agent.x += move_dist
                    agent.last_action = 'E%i' % (move_dist)
                elif action == ind_start + 2:
                    agent.y -= move_dist
                    agent.last_action = 'S%i' % (move_dist)
                elif action == ind_start + 3:
                    agent.x -= move_dist
                    agent.last_action = 'W%i' % (move_dist)
            if action >= self.n_actions - 2:
                # building interaction actions
                act_bld = list(filter(lambda x: x.ent_type == 'building',
                                      self._vis_objs(agent.x, agent.y, 1)))[0]
                if action == self.n_actions - 2:
                    act_bld.reduce_fire(agent.fire_speed)
                    agent.fire_actions += 1
                    agent.last_action = 'F'
                elif action == self.n_actions - 1:
                    act_bld.build(agent.build_speed)
                    agent.build_actions += 1
                    agent.last_action = 'B'

        for building in self.buildings:
            bld_info = building.step()
            total_reward += self.reward_config['health_delta_mult'] * bld_info['buildings_health_delta']
            total_reward += self.reward_config['complete'] * bld_info['buildings_completed']
            total_reward += self.reward_config['burned_down'] * bld_info['buildings_burned']
            total_reward += self.reward_config['extinguish'] * bld_info['fires_extinguished']

            for k, v in bld_info.items():
                self.ep_info[k] += v

        # check for collisions
        for ia, agent in enumerate(self.agents):
            for oa in range(ia + 1, self.n_agents):
                other = self.agents[oa]
                dist = abs(agent.x - other.x) + abs(agent.y - other.y)
                if dist == 0:
                    # randomly choose agent to take the disputed position
                    chosen_agent = np.random.choice([agent, other])
                    (chosen_agent.x, chosen_agent.y) = (chosen_agent.prev_x,
                                                        chosen_agent.prev_y)
        self._update_grid_pos()

        self.time += 1

        info = {}
        done = False
        if all((b.complete or b.burned_down) for b in self.buildings):
            done = True
        elif self.time >= self.episode_limit:
            done = True
            info["episode_limit"] = True

        if done:
            info["prop_buildings_completed"] = sum(b.complete for b in self.buildings) / len(self.buildings)
            info["solved"] = int(all(b.complete for b in self.buildings))
            if info["solved"] == 1:
                total_reward += self.reward_config['solved']
            for k, v in self.ep_info.items():
                info[k] = v

            if self.track_ac_type:
                ac_types = np.zeros((len(AGENT_TYPES), 2))
                for agent in self.agents:
                    ac_types[agent.ent_id - N_BUILDING_TYPES, 0] += agent.fire_actions
                    ac_types[agent.ent_id - N_BUILDING_TYPES, 1] += agent.build_actions
                total_acs = ac_types.sum(axis=1, keepdims=True)
                total_acs[total_acs == 0] = 1
                ac_types /= total_acs
                info['action_types'] = ac_types
        total_reward = (total_reward * self.reward_scale) / self.max_reward
        return total_reward, done, info

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"entity_shape": self.get_entity_size(),
                    "n_actions": self.n_actions,
                    "n_agents": self.max_n_agents,
                    "n_entities": self.max_n_agents + self.max_n_buildings,
                    "episode_limit": self.episode_limit}
        return env_info

    def render(self, mode='human', close=False, verbose=True):
        self.init_render()
        self._render.new_frame()

        agent_type_colors = ['red', 'blue', 'green']
        build_type_colors = ['green', 'yellow']
        action_dict = {'F': 'blue', 'B': 'black'}

        bg = plt.Rectangle((0, 0), MAP_SIZE + 1, MAP_SIZE + 1,
                           linewidth=5, edgecolor='black', facecolor='white',
                           fill=True, zorder=1.3, alpha=1.0)
        self._render.add_artist(bg)
        for agent in self.agents:
            if agent.sight_range is not None:
                agent_view = plt.Rectangle((agent.x - agent.sight_range, agent.y - agent.sight_range), agent.sight_range * 2 + 1, agent.sight_range * 2 + 1,
                                           linewidth=5, edgecolor='black', facecolor='black', fill=True,
                                           zorder=1.4, alpha=0.1)
                self._render.add_artist(agent_view)

            agent_sq = plt.Rectangle((agent.x, agent.y), 1, 1,
                                     facecolor=agent_type_colors[agent.ent_id - N_BUILDING_TYPES],
                                     zorder=1.5, alpha=1.0)
            self._render.add_artist(agent_sq)

            if agent.last_action in action_dict:
                agent_action = plt.Rectangle((agent.x, agent.y), 1, 1,
                                             linewidth=5, edgecolor=action_dict[agent.last_action],
                                             fill=False,
                                             zorder=1.6, alpha=1.0)
                self._render.add_artist(agent_action)

        for bld in self.buildings:
            build_sq = plt.Rectangle((bld.x, bld.y), 1, 1,
                                     edgecolor=build_type_colors[bld.ent_id], fill=False, linewidth=5,
                                     zorder=1.6, alpha=1.0)
            self._render.add_artist(build_sq)
            if bld.complete:
                build_comp = plt.Rectangle((bld.x - 0.1, bld.y - 0.1), 1 + 0.2, 1 + 0.2,
                                           edgecolor='black', fill=False, linewidth=5,
                                           zorder=1.5, alpha=1.0)
                self._render.add_artist(build_comp)
            if bld.burned_down:
                build_burned = plt.Rectangle((bld.x - 0.1, bld.y - 0.1), 1 + 0.2, 1 + 0.2,
                                             edgecolor='red', fill=False, linewidth=5,
                                             zorder=1.5, alpha=1.0)
                self._render.add_artist(build_burned)
            if bld.health > 0.0:
                build_health = plt.Rectangle((bld.x, bld.y), 0.5, bld.health,
                                             facecolor='black',
                                             zorder=1.5, alpha=1.0)
                self._render.add_artist(build_health)
            if bld.fire_strength > 0.0:
                fire_stat = plt.Rectangle((bld.x + 0.5, bld.y), 0.5, bld.fire_strength,
                                          facecolor='red',
                                          zorder=1.5, alpha=1.0)
                self._render.add_artist(fire_stat)
            build_desc = plt.Annotation("%3.0f%%, %3.0f%%" % (bld.health * 100, bld.fire_strength * 100),
                                        (bld.x - 0.1, bld.y - 0.4))
            self._render.add_artist(build_desc)

        # self._render.fig.clear()
        image = self._render.draw()
        # self._render.render(image)
        return image

    def init_render(self):
        if self._render is None:
            self._render = Render()
        return self
