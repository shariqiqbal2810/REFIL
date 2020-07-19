import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch, repeat_batch=1):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1, 1),
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                     batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                    (entities[:, 1:],
                     batch["entity_mask"][:, 1:]))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()
        # for t in range(batch.max_seq_length):
        #     agent_outs = self.mac.forward(batch, t=t)
        #     mac_out.append(agent_outs)
        # mac_out = th.stack(mac_out, dim=1)  # Concat over time
        if 'imagine' in self.args.agent:
            all_mac_out, groups = self.mac.forward(batch, t=None, imagine=True)
            # Pick the Q-Values for the actions taken by each agent
            rep_actions = actions.repeat(3, 1, 1, 1)
            all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

            mac_out, moW, moI = all_mac_out.chunk(3, dim=0)
            chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
            if not self.args.mix_imagined:
                caq_imagine = caqW + caqI
            else:
                caq_imagine = th.cat([caqW, caqI], dim=2)
        else:
            mac_out = self.mac.forward(batch, t=None)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out = self.target_mac.forward(batch, t=None)
        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if 'imagine' in self.args.agent:
                if not self.args.mix_imagined:
                    mix_ins, targ_mix_ins = self._get_mixer_ins(batch,
                                                                repeat_batch=2)
                    mixer_qvals = th.cat([chosen_action_qvals, caq_imagine], dim=0)
                    chosen_action_qvals = self.mixer(mixer_qvals, mix_ins)
                    chosen_action_qvals, caq_imagine = chosen_action_qvals.chunk(2, dim=0)
                else:
                    mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                    chosen_action_qvals = self.mixer(chosen_action_qvals,
                                                     mix_ins)
                    # don't need last timestep
                    groups = [gr[:, :-1] for gr in groups]
                    caq_imagine = self.mixer(caq_imagine, mix_ins,
                                             imagine_groups=groups)
            else:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if 'imagine' in self.args.agent:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            if 'imagine' in self.args.agent:
                self.logger.log_stat("im_loss", im_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            if self.args.gated:
                self.logger.log_stat("gate", self.mixer.gate.cpu().item(), t_env)
            if batch.max_seq_length == 2:
                # We are in a 1-step env. Calculate the max Q-Value for logging
                max_agent_qvals = mac_out_detach[:,0].max(dim=2, keepdim=True)[0]
                max_qtots = self.mixer(max_agent_qvals, batch["state"][:,0])
                self.logger.log_stat("max_qtot", max_qtots.mean().item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if hasattr(self, "imagine_mixer"):
            self.imagine_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if hasattr(self, "imagine_mixer"):
            th.save(self.imagine_mixer.state_dict(), "{}/imagine_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path, evaluate=False):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if not evaluate:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            if hasattr(self, "imagine_mixer"):
                self.imagine_mixer.load_state_dict(th.load("{}/imagine_mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
