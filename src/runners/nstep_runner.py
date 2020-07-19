import torch as th
import numpy as np

from controllers import REGISTRY as c_REGISTRY
# from components.scheme import Scheme
# from components.episode_buffer_old import BatchEpisodeBuffer
from components.env_stats_aggregators import REGISTRY as envstats_REGISTRY
# from components.transforms_old import _join_dicts, _underscore_to_cap, _copy_remove_keys, _make_logging_str, _seq_mean
from copy import deepcopy
from envs import REGISTRY as env_REGISTRY
from functools import partial
import queue
from threading import Thread
from torch import multiprocessing as mp

# class Runner(object):
#
#     def __init__(self, env, agents, action_selector, args):
#         self.args = args
#
#     def run(self):
#         raise NotImplementedError

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class NStepRunner():

    def __init__(self, multiagent_controller=None,
                   args=None,
                   logging_struct=None,
                   data_scheme=None,
                   **kwargs):

        # assert that number of processes/threads/loop workers aligns with batch size!

        #self.n_agents = args.n_agents
        self.contiguous = True # may want to parametrize some day
        self.env = args.env
        self.args=args
        self.n_loops_per_thread_or_sub_or_main_process = self.args.n_loops_per_thread_or_sub_or_main_process
        self.n_threads_per_subprocess_or_main_process = self.args.n_threads_per_subprocess_or_main_process
        self.n_subprocesses = self.args.n_subprocesses
        self.queue = None
        self.subprocesses = None
        self.T_env = 0
        self.max_t_episode = args.t_max
        self.batch_size = kwargs.get("batch_size", self.args.batch_size_run)
        self.transition_buffer_is_cuda = kwargs.get("transition_buffer_is_cuda", False) # only makes sense for subprocesses
        self.msg_response_timeout = kwargs.get("msg_response_timeout", 3) # default: 3s, might increase for really slow envs
        self.logging_struct = logging_struct

        # some convenience configuration
        if self.n_subprocesses == 0 and self.n_threads_per_subprocess_or_main_process == 0:
            if self.n_loops_per_thread_or_sub_or_main_process == 0:
                self.n_loops_per_thread_or_sub_or_main_process = self.batch_size
            else:
                assert self.n_loops_per_thread_or_sub_or_main_process == self.batch_size, "if no threads or subprocesses, number of loop workers has to align with batch size"

        # Set up and create the environments
        # May have several different envs etc etc, but all need to have same output dims etc (for now)!
        envs_fn = env_REGISTRY[args.env]
        envs_fn = partial(envs_fn, env_args = self.args.env_args)

        if self.n_subprocesses > 0:
            assert self.batch_size % self.n_subprocesses == 0, "runner batch size has to be multiple of number of sub-processes for now!"
        self._setup_envs(envs_fn)


        # retrieve initial environment setup info
        self.env_setup_info = self._exch_msgs(ids=range(0, self.batch_size), msgs=["SCHEME"]*self.batch_size)
        assert all([self.env_setup_info[0]["obs_size"] == _env_setup["obs_size"] for _, _env_setup in self.env_setup_info.items()]), \
                "observation sizes have to be uniform across environments!"
        assert all([self.env_setup_info[0]["state_size"] == _env_setup["state_size"] for _, _env_setup in self.env_setup_info.items()]), \
                "observation sizes have to be uniform across environments!"

        self.env_obs_size = self.env_setup_info[0]["obs_size"]
        self.env_state_size = self.env_setup_info[0]["state_size"]
        self.env_episode_limit = self.env_setup_info[0]["episode_limit"] if self.env_setup_info[0]["episode_limit"] != 0 \
                                 else self.args.env_args["episode_limit"]
        self.n_agents = self.env_setup_info[0]["n_agents"]
        self.n_actions = self.env_setup_info[0]["n_actions"]

        # After we have set up the environments, we can now construct the runner scheme
        self._setup_data_scheme(data_scheme)

        # set up env stats aggregation module
        if hasattr(args, "env_stats_aggregator") and args.env_stats_aggregator is not None:
            self.env_stats_aggregator = envstats_REGISTRY[args.env_stats_aggregator]()
        else:
            self.env_stats_aggregator = None

        # set up multiagent controller
        if multiagent_controller is None:
            self.multiagent_controller = c_REGISTRY[args.multiagent_controller](runner=self,
                                                                                n_agents=self.n_agents,
                                                                                n_actions=self.n_actions,
                                                                                args=self.args,
                                                                                logging_struct=self.logging_struct)
            self.multiagent_controller.create_model(self.data_scheme)
        else:
            self.multiagent_controller = multiagent_controller


        # finally, set up memory buffers
        # This buffer contains the whole output of the current runner's run (i.e. a batch of episodes)
        self.episode_buffer = BatchEpisodeBuffer(data_scheme=self.data_scheme,
                                                 n_bs=self.batch_size,
                                                 n_t=self.env_episode_limit+1,
                                                 n_agents=self.n_agents,
                                                 is_cuda=self.args.use_cuda,
                                                 is_shared_mem=False)

        # We set up a single-transition buffer that is shared among the processes and serves to store the latest transitions
        # The reason for why we set up a separate transition buffer is that if cuda is not enabled, we cannot use shared tensors
        # but have to rely on a special shared array structure that is subsequently being copied to GPU.
        # Is is also not clear whether it is faster to let each worker copy the results directly to GPU, or batch-copy them to GPU later
        # TODO: if is_shared_mem=True but is_cuda is False, create shared numpy array instead! (and restrict functionality of CTSB object)
        self.transition_buffer = BatchEpisodeBuffer(data_scheme=self.data_scheme,
                                                    n_bs=self.batch_size,
                                                    n_t=1,
                                                    n_agents=self.n_agents,
                                                    is_cuda=self.transition_buffer_is_cuda,
                                                    is_shared_mem=True)

        pass

    def _setup_data_scheme(self, data_scheme):
        if data_scheme is not None:
            self.data_scheme = data_scheme
        else:
            self.data_scheme = Scheme([dict(name="observations",
                                            shape=(self.env_obs_size,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.float32,
                                            missing=np.nan),
                                       dict(name="state",
                                            shape=(self.env_state_size,),
                                            dtype=np.float32,
                                            missing=np.nan,
                                            size=self.env_state_size),
                                       dict(name="actions",
                                            shape=(1,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.int32,
                                            missing=-1,),
                                       dict(name="avail_actions",
                                            shape=(self.n_actions,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.int32,
                                            missing=-1,),
                                       dict(name="reward",
                                            shape=(1,),
                                            dtype=np.float32,
                                            missing=np.nan),
                                       dict(name="agent_id",
                                            shape=(1,),
                                            dtype=np.int32,
                                            select_agent_ids=range(0, self.n_agents),
                                            missing=-1),
                                       dict(name="epsilon",
                                            shape=(1,),
                                            dtype=np.float32,
                                            missing=np.nan,
                                            switch=self.args.obs_epsilon),
                                       dict(name="policies",
                                            shape=(self.n_actions,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.float32,
                                            missing=np.nan,
                                            switch=self.args.action_selector in ["multinomial"]),
                                       dict(name="q_values",
                                            shape=(self.n_actions,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.float32,
                                            missing=np.nan,
                                            switch=not self.args.action_selector in ["multinomial"]),
                                       dict(name="v_values",
                                            shape=(self.n_actions,),
                                            select_agent_ids=range(0, self.n_agents),
                                            dtype=np.float32,
                                            missing=np.nan,
                                            switch=not self.args.action_selector in ["multinomial"]),
                                       dict(name="terminated",
                                            shape=(1,),
                                            dtype=np.bool,
                                            missing=False),
                                       dict(name="truncated",
                                            shape=(1,),
                                            dtype=np.bool,
                                            missing=False),
                                       dict(name="reset",
                                            shape=(1,),
                                            dtype=np.bool,
                                            missing=False),
                                       ]).agent_flatten()


    def share_memory(self):
        self.multiagent_controller.share_memory()
        pass

    def _setup_envs(self, envs_fn):
        """
        setup subprocess/thread/loop tree
        receives the appropriate amount of environment creation functions in a per-subprocess vector
        """

        if self.n_subprocesses > 0:
            self.in_queues = [mp.Queue() for _ in range(self.n_subprocesses)] # need one in_queue per subprocess as need to keep envs subprocess-local
        else:
            self.in_queues = mp.Queue()

        self.out_queue = mp.Queue() # out_queue can be shared across subprocesses
        self.T_queue = mp.Queue()  # Maybe create outside of loop but should be super fast anyway...

        if self.n_subprocesses > 0:
            n_items_per_subproc = int(self.batch_size / self.n_subprocesses)
            self.subprocesses = [mp.Process(target=self._subprocess_worker,
                                            args=(dict(subproc_id=_i,
                                                       envs_fn=CloudpickleWrapper(partial(envs_fn, subproc_id=_i)),
                                                       in_queue=in_queue,
                                                       out_queue=self.out_queue,
                                                       T_queue=self.T_queue,
                                                       n_loops_per_thread_or_sub_or_main_process=self.n_loops_per_thread_or_sub_or_main_process,
                                                       n_threads_per_subprocess_or_main_process=self.n_threads_per_subprocess_or_main_process,
                                                       n_subprocesses=self.n_subprocesses,
                                                       thread_worker_fn=CloudpickleWrapper(partial(self._thread_worker,
                                                                                                   subproc_id=_i)),
                                                       buffer_insert_fn=CloudpickleWrapper(partial(self.buffer_insert,
                                                                                                   subproc_id_set=range(n_items_per_subproc*_i,
                                                                                                                        n_items_per_subproc*(_i+1)),
                                                                                                   )),
                                                       loop_worker_fn=CloudpickleWrapper(partial(self._loop_worker,
                                                                                                   subproc_id=_i,
                                                                                                   args=self.args)),
                                                       ),)) for _i, in_queue in zip(range(self.n_subprocesses), self.in_queues)]
            for p in self.subprocesses:
                p.daemon = True
                p.start()

        elif self.n_threads_per_subprocess_or_main_process > 0:
            self._subprocess_worker(0, envs_fn, self.in_queues, self.out_queue, self.T_queue, args=self.args)
        else:
            # create the environments straight here and add to self
            self.envs = [envs_fn(bs_id=_i) for _i in range(self.batch_size)]
        pass

    @staticmethod
    def _subprocess_worker(kwargs):
        """

        environment objects are kept at a per-subprocess (or, if no subprocesses, main process) scope
        """
        subproc_id = kwargs["subproc_id"]
        envs_fn = kwargs["envs_fn"]
        in_queue = kwargs["in_queue"]
        out_queue = kwargs["out_queue"]
        T_queue = kwargs["T_queue"]
        n_threads_per_subprocess_or_main_process = kwargs["n_threads_per_subprocess_or_main_process"]
        n_subprocesses = kwargs["n_subprocesses"]
        n_loops_per_thread_or_sub_or_main_process = kwargs["n_loops_per_thread_or_sub_or_main_process"]
        thread_worker_fn = kwargs["thread_worker_fn"]
        loop_worker_fn = kwargs["loop_worker_fn"]
        buffer_insert_fn = kwargs["buffer_insert_fn"]
        args = kwargs.get("args", None)

        if isinstance(envs_fn, CloudpickleWrapper):
            envs_fn = envs_fn.x

        envs = {}
        if n_threads_per_subprocess_or_main_process > 0:
            workers = []
            for _i in range(n_threads_per_subprocess_or_main_process):
                [envs.update(_j=envs_fn(thread_id=_i,
                                        loop_id=_j,
                                        bs_id=subproc_id*(n_threads_per_subprocess_or_main_process + n_loops_per_thread_or_sub_or_main_process)
                                              + _i*n_loops_per_thread_or_sub_or_main_process
                                              + _j)) for _j in range(subproc_id*n_loops_per_thread_or_sub_or_main_process,
                                                                                       (subproc_id+1)*n_loops_per_thread_or_sub_or_main_process)] # dimensions: thread x loop
                workers.append(Thread(target=thread_worker_fn, args=(dict(envs=envs,
                                                                          in_queue=in_queue,
                                                                          out_queue=out_queue,
                                                                          T_queue=T_queue,
                                                                          n_threads_per_subprocess_or_main_process=n_threads_per_subprocess_or_main_process,
                                                                          n_subprocesses=n_subprocesses,
                                                                          n_loops_per_thread_or_sub_or_main_process=n_loops_per_thread_or_sub_or_main_process,
                                                                          loop_worker_fn=partial(loop_worker_fn, args=args),
                                                                          buffer_insert_fn=buffer_insert_fn))))
                workers[-1].setDaemon(True)
                workers[-1].start()

            if n_subprocesses > 0:
                for _w in workers:
                    _w.join()

            return "TERMINATE"
        else:
            if n_loops_per_thread_or_sub_or_main_process > 0:
                envs = [envs_fn(loop_id=_j, bs_id=subproc_id*n_loops_per_thread_or_sub_or_main_process
                                                  + _j) for _j in range(n_loops_per_thread_or_sub_or_main_process)] # dimensions: loop
            else:
                envs = [envs_fn(loop_id=0, bs_id=subproc_id)]

            if isinstance(thread_worker_fn, CloudpickleWrapper):
                thread_worker_fn = thread_worker_fn.x
            thread_worker_fn({**dict(envs=envs, in_queue=in_queue, out_queue=out_queue, T_queue=T_queue), **kwargs}) # executed in main context of subprocess
            pass
        pass

    @staticmethod
    def _thread_worker(kwargs, subproc_id=None):
        envs = kwargs["envs"]
        in_queue = kwargs["in_queue"]
        out_queue = kwargs["out_queue"]
        n_threads_per_subprocess_or_main_process = kwargs["n_threads_per_subprocess_or_main_process"]
        n_subprocesses = kwargs["n_subprocesses"]
        n_loops_per_thread_or_sub_or_main_process = kwargs["n_loops_per_thread_or_sub_or_main_process"]
        loop_worker_fn = kwargs["loop_worker_fn"]
        buffer_insert_fn = kwargs["buffer_insert_fn"]

        if isinstance(loop_worker_fn, CloudpickleWrapper):
            loop_worker_fn = loop_worker_fn.x

        if isinstance(buffer_insert_fn, CloudpickleWrapper):
            buffer_insert_fn = buffer_insert_fn.x

        while True:
            terminated_counter = 0
            if n_loops_per_thread_or_sub_or_main_process > 0:
                    ret = loop_worker_fn(envs=envs,
                                         in_queue=in_queue,
                                         out_queue=out_queue,
                                         buffer_insert_fn=buffer_insert_fn)
                    if ret == "TERMINATED":
                        terminated_counter += 1
                    if terminated_counter == n_loops_per_thread_or_sub_or_main_process:
                        break
            else:
                ret = loop_worker_fn(envs=envs,
                                     in_queue=in_queue,
                                     out_queue=out_queue,
                                     buffer_insert_fn=buffer_insert_fn)
                if ret == "TERMINATED":
                    break
                if n_subprocesses == 0 and n_threads_per_subprocess_or_main_process == 0:
                    break

        return "TERMINATE"

    @staticmethod
    def buffer_insert(id, subproc_id_set, buffer, column_scheme, data_dict):
        """
        high-performance insertion routine for transition buffer
        """

        #    Do some rudimentary memory access violation checks here
        assert id in subproc_id_set, \
            "ACCESS VIOLATION: buffer not accessible at id {} from subproc_id_set {}".format(id, subproc_id_set)

        for _k, _v in data_dict.items():
            if isinstance(_v, np.ndarray):
                _v = th.FloatTensor(_v)
            elif isinstance(_v, (list, tuple)):
                _v = th.FloatTensor(_v)
            elif isinstance(_v, bool):
                _v = int(_v)
            buffer._transition[id, 0, column_scheme[_k][0]:column_scheme[_k][1]] = _v
        pass
    def terminate(self):
        """
        clean termination of all threads
        """
        self._exch_msgs(["TERMINATE"]*self.batch_size)
        return 0

    @staticmethod
    def _loop_worker(envs,
                     in_queue,
                     out_queue,
                     buffer_insert_fn,
                     subproc_id=None,
                     args=None,
                     msg=None):

        if in_queue is None:
            id, chosen_actions, output_buffer, column_scheme = msg
            env_id = id
        else:
            id, chosen_actions, output_buffer, column_scheme = in_queue.get() # timeout=1)
            env_id_offset = len(envs) * subproc_id  # TODO: Adjust for multi-threading!
            env_id = id - env_id_offset

        _env = envs[env_id]

        if chosen_actions == "SCHEME":
            env_dict = dict(obs_size=_env.get_obs_size(),
                            state_size=_env.get_state_size(),
                            episode_limit=_env.episode_limit,
                            n_agents = _env.n_agents,
                            n_actions=_env.get_total_actions())

            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "RESET":
            _env.reset() # reset the env!

            # perform environment steps and insert into transition buffer
            observations = _env.get_obs()
            state = _env.get_state()
            avail_actions = _env.get_avail_actions()
            ret_dict = dict(state=state)  # TODO: Check that env_info actually exists
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="RESET DONE"))
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        elif chosen_actions == "STATS":
            env_stats = _env.get_stats()
            env_dict = dict(env_stats=env_stats)
            # Send results back
            ret_msg = dict(id=id, payload=env_dict)
            if out_queue is None:
                return ret_msg
            out_queue.put(ret_msg)
            return

        else:
            reward, terminated, env_info = \
                _env.step([int(_i) for _i in chosen_actions])

            # perform environment steps and add to transition buffer
            observations = _env.get_obs()
            state = _env.get_state()
            avail_actions = _env.get_avail_actions()
            terminated = terminated
            truncated = env_info.get("episode_limit", False)

            if env_info.get("episode_limit", False):
                # The episode terminated because of a time limit
                terminated = False
            env_finished = terminated or truncated

            ret_dict = dict(state=state,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            )
            for _i, _obs in enumerate(observations):
                ret_dict["observations__agent{}".format(_i)] = observations[_i]
            for _i, _obs in enumerate(observations):
                ret_dict["avail_actions__agent{}".format(_i)] = avail_actions[_i]

            buffer_insert_fn(id=id, buffer=output_buffer, data_dict=ret_dict, column_scheme=column_scheme)

            # Signal back that queue element was finished processing
            ret_msg = dict(id=id, payload=dict(msg="STEP DONE", terminated=env_finished))
            if out_queue is None:
                return ret_msg
            else:
                out_queue.put(ret_msg)
            return

        return


    def _main_thread_worker(self, ids, msgs):
        res = []
        for id, msg in zip(ids, msgs):
            res.append(self._loop_worker(envs=self.envs,
                                         in_queue=None,
                                         out_queue=None,
                                         buffer_insert_fn=partial(self.buffer_insert,
                                                                  subproc_id_set=[id],),
                                         args=self.args,
                                         msg=msg))
            # column_scheme=self.transition_buffer.columns._transition if hasattr(self, "transition_buffer") else None,),
        return res

    def _exch_msgs(self, ids, msgs):
        """
        helper function that sends ordered messages to all envs specified by their ids, and returns the answers received in ordered form.
        """

        # set up messages with process ids and corresponding out_buffer objects
        # NOTE: pytorch multiprocessing best practices recommend always sharing pytorch tensors via queues, rather than at start-up
        if hasattr(self, "transition_buffer") and self.transition_buffer is not None:
            output_buffers = (self.transition_buffer.data for _ in range(len(ids)))
            column_schemes = (self.transition_buffer.columns._transition for _ in range(len(ids)))
        else:
            output_buffers = (None for _ in range(len(ids)))
            column_schemes = (None for _ in range(len(ids)))
        msgs_with_ids = list(zip(ids, msgs, output_buffers, column_schemes))

        if self.n_subprocesses > 0:
            n_items_per_subproc = int(self.batch_size / self.n_subprocesses)

            # calculate which id is referring to which subprocess
            subproc_ids = [ _id // n_items_per_subproc   for _id in ids]
            [self.in_queues[_subproc_id].put(_msg) for _subproc_id, _msg in zip(subproc_ids, msgs_with_ids)]

        elif self.n_subprocesses == 0 and self.n_threads_per_subprocess_or_main_process > 0:
            list(map(self.in_queues.put, zip(range(self.batch_size), msgs)))

        elif self.n_subprocesses == 0 and self.n_threads_per_subprocess_or_main_process == 0:

            # in this case, there are simply no background workers who could chip away at the queue -
            # we need to call function explicitely within the current scope therefore!
            res = self._main_thread_worker(ids=ids,
                                           msgs=msgs_with_ids)
            ret = {_r["id"]:_r["payload"] for _r in res}
            return ret

        # read out results from queue
        ret = {}
        for _i in range(len(ids)):
            _ret = self.out_queue.get()
            ret[_ret["id"]] = _ret["payload"]
        return ret

    def reset_envs(self):
        return self._exch_msgs(ids=range(0, self.batch_size),
                               msgs=["RESET"] * self.batch_size)

    def reset(self):

        # reset episode time step counter
        self.t_episode = -1

        # can flush the memory buffer now, that's fine
        self.episode_buffer.flush()

        # flush the transition buffer BEFORE resetting the envs, as these will write into it during reset.
        self.transition_buffer.flush()

        # reset environments
        self.reset_envs()

        # copy initial transition into episode buffer
        self.episode_buffer.insert(self.transition_buffer, t_ids=0, bs_ids=list(range(0, self.batch_size)))

        # c = self.transition_buffer.to_pd() #DEBUG

        # re-initialize the hidden states
        self.hidden_states, self.hidden_states_format = self.multiagent_controller.generate_initial_hidden_states(self.batch_size)

        # re-initialize the environment states (these are strictly required NOT to terminate at reset)
        self.envs_terminated = [False]*self.batch_size
        self.envs_truncated = [False]*self.batch_size

        pass

    def step(self, actions, ids):
        """

        """
        selected_actions_msgs = [actions[:, _id, 0, 0].tolist() for _id in range(len(ids))]
        ret = self._exch_msgs(msgs=selected_actions_msgs, ids=[_b for _b in ids])
        return ret

    def run(self, test_mode):
        self.test_mode = test_mode

        # don't reset at initialization as don't have access to hidden state size then
        self.reset()

        terminated = False
        while not terminated:
            # increase episode time counter
            self.t_episode += 1

            # retrieve ids of all envs that have not yet terminated.
            # NOTE: for efficiency reasons, will perform final action selection in terminal state
            ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
            ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                                                if self.episode_buffer.is_cuda \
                                                else th.LongTensor(ids_envs_not_terminated)


            if self.t_episode > 0:

                # flush transition buffer before next step
                self.transition_buffer.flush()

                # get selected actions from last step
                selected_actions, selected_actions_tformat = self.episode_buffer.get_col(col="actions",
                                                                                         t=self.t_episode-1,
                                                                                         agent_ids=list(range(self.n_agents))
                                                                                         )

                ret = self.step(actions=selected_actions[:, ids_envs_not_terminated_tensor.cuda()
                                                             if selected_actions.is_cuda else ids_envs_not_terminated_tensor.cpu(), :, :],
                                ids=ids_envs_not_terminated)

                # retrieve ids of all envs that have not yet terminated.
                # NOTE: for efficiency reasons, will perform final action selection in terminal state
                ids_envs_not_terminated = [_b for _b in range(self.batch_size) if not self.envs_terminated[_b]]
                ids_envs_not_terminated_tensor = th.cuda.LongTensor(ids_envs_not_terminated) \
                    if self.episode_buffer.is_cuda \
                    else th.LongTensor(ids_envs_not_terminated)

                # update which envs have terminated
                for _id, _v in ret.items():
                    self.envs_terminated[_id] = _v["terminated"]

                # insert new data in transition_buffer into episode buffer (NOTE: there's a good reason for why processes
                # don't write directly into the episode buffer)
                self.episode_buffer.insert(self.transition_buffer,
                                           bs_ids=list(range(self.batch_size)),
                                           t_ids=self.t_episode,
                                           bs_empty=[_i for _i in range(self.batch_size) if _i not in ids_envs_not_terminated])

                # update episode time counter
                if not self.test_mode:
                    self.T_env += len(ids_envs_not_terminated)

            #a = self.episode_buffer.to_pd()
            # generate multiagent_controller inputs for policy forward pass
            multiagent_controller_inputs, \
            multiagent_controller_inputs_tformat = self.episode_buffer.view(dict_of_schemes=self.multiagent_controller.joint_scheme_dict,
                                                                            #scheme=self.multiagent_controller.schemes["joint"],
                                                                            to_cuda=self.args.use_cuda,
                                                                            to_variable=True,
                                                                            bs_ids=ids_envs_not_terminated,
                                                                            t_id=self.t_episode,
                                                                            fill_zero=True, # TODO: DEBUG!!!
                                                                            )

            # retrieve avail_actions from episode_buffer
            avail_actions, avail_actions_format = self.episode_buffer.get_col(bs=ids_envs_not_terminated,
                                                                              col="avail_actions",
                                                                              t = self.t_episode,
                                                                              agent_ids=list(range(self.n_agents)))

            # a = multiagent_controller_inputs[list(multiagent_controller_inputs.keys())[0]].to_pd()
            # forward-pass to obtain current agent policy
            if isinstance(self.hidden_states, dict):
                hidden_states = {_k:_v[:, ids_envs_not_terminated_tensor, :, :] for _k, _v in self.hidden_states.items()}
            else:
                hidden_states = self.hidden_states[:, ids_envs_not_terminated_tensor, :,:]


            multiagent_controller_outputs, multiagent_controller_outputs_tformat = \
                self.multiagent_controller.get_outputs(inputs=multiagent_controller_inputs,
                                                       hidden_states=hidden_states,
                                                       avail_actions=avail_actions,
                                                       tformat=multiagent_controller_inputs_tformat,
                                                       test_mode=test_mode,
                                                       info=None)

            if isinstance(multiagent_controller_outputs["hidden_states"], dict):
                for _k, _v in multiagent_controller_outputs["hidden_states"].items():
                    self.hidden_states[_k][:, ids_envs_not_terminated_tensor, :, :] = _v
            else:
                self.hidden_states[:, ids_envs_not_terminated_tensor, :, :] = multiagent_controller_outputs["hidden_states"]

            selected_actions, action_selector_outputs, selected_actions_format = \
                self.multiagent_controller.select_actions(inputs=multiagent_controller_outputs,
                                                          avail_actions=avail_actions,
                                                          tformat=avail_actions_format,
                                                          info=dict(T_env=self.T_env),
                                                          test_mode=test_mode)

            # TODO: can encapsulate this in some common function
            if isinstance(action_selector_outputs, list):
                for _sa in action_selector_outputs:
                    self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col=_sa["name"],
                                            t=self.t_episode,
                                            agent_ids=_sa.get("select_agent_ids", None),
                                            data=_sa["data"])

            else:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col=self.multiagent_controller.agent_output_type,
                                            t=self.t_episode,
                                            agent_ids=list(range(self.n_agents)),
                                            data=action_selector_outputs)

            # write selected actions to episode_buffer
            if isinstance(selected_actions, list):
               for _sa in selected_actions:
                   self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                               col=_sa["name"],
                                               t=self.t_episode,
                                               agent_ids=_sa.get("select_agent_ids", None),
                                               data=_sa["data"])
            else:
                self.episode_buffer.set_col(bs=ids_envs_not_terminated,
                                            col="actions",
                                            t=self.t_episode,
                                            agent_ids=list(range(self.n_agents)),
                                            data=selected_actions)

            # keep a copy of selected actions explicitely in transition_buffer device context
            #self.selected_actions = selected_actions.cuda() if self.transition_buffer.is_cuda else selected_actions.cpu()

            #Check for termination conditions
            #Check for runner termination conditions
            if self.t_episode == self.max_t_episode:
                terminated = True
            # Check whether all envs have terminated
            if all(self.envs_terminated):
                terminated = True
            # Check whether envs may have failed to terminate
            if self.t_episode == self.env_episode_limit+1 and not terminated:
                assert False, "Envs seem to have failed returning terminated=True, thus not respecting their own episode_limit. Please fix envs."

            pass

        # calculate episode statistics
        self._add_episode_stats(T_env=self.T_env)
        # a = self.episode_buffer.to_pd()
        return self.episode_buffer

    def _add_episode_stats(self, T_env):

        test_suffix = "" if not self.test_mode else "_test"
        if self.env_stats_aggregator is not None:
            # receive episode stats from envs
            stats_msgs = ["STATS"]*self.batch_size
            env_stats = self._exch_msgs(msgs=stats_msgs, ids=range(self.batch_size))
            self.env_stats_aggregator.aggregate(stats=[env_stats[_id]["env_stats"] for _id in range(self.batch_size)],
                                                add_stat_fn=partial(self._add_stat, T_env=T_env, suffix=test_suffix))

        self._add_stat("T_env", T_env, T_env=T_env, suffix=test_suffix)
        self._add_stat("episode_reward", np.mean(self.episode_buffer.get_stat("reward_sum", bs_ids=None)), T_env=T_env,
                       suffix=test_suffix)
        self._add_stat("episode_length", np.mean(self.episode_buffer.get_stat("episode_length", bs_ids=None)),
                       T_env=T_env, suffix=test_suffix)

        if self.test_mode:
            if self.args.save_episode_samples:
                assert self.args.use_hdf_logger, "use_hdf_logger needs to be enabled if episode samples are to be stored!"
                self.logging_struct.hdf_logger.log("_test", self.episode_buffer, self.T_env)

        pass

    def _add_stat(self, name, value, T_env, suffix=""):
        name += suffix

        if isinstance(value, np.ndarray) and value.size == 1:
            value = float(value)

        if not hasattr(self, "_stats"):
            self._stats = {}

        if name not in self._stats:
            self._stats[name] = []
            self._stats[name+"_T_env"] = []
        self._stats[name].append(value)
        self._stats[name+"_T_env"].append(self.T_env)

        if hasattr(self, "max_stats_len") and len(self._stats) > self.max_stats_len:
            self._stats[name].pop(0)
            self._stats[name+"_T_env"].pop(0)

        # log to sacred if enabled
        if hasattr(self.logging_struct, "sacred_log_scalar_fn"):
            self.logging_struct.sacred_log_scalar_fn(key=_underscore_to_cap(name), val=value)

        # log to tensorboard if enabled
        if hasattr(self.logging_struct, "tensorboard_log_scalar_fn"):
            self.logging_struct.tensorboard_log_scalar_fn(_underscore_to_cap(name), value, T_env)

        # log to hdf if enabled
        if hasattr(self.logging_struct, "hdf_logger"):
            self.logging_struct.hdf_logger.log(_underscore_to_cap(name), value, T_env)

        return

    def log(self,log_directly = True):
        """
        Each l0earner has it's own logging routine, which logs directly to the python-wide logger if log_directly==True,
        and returns a logging string otherwise

        Logging is triggered in run.py
        """
        test_suffix = "" if not self.test_mode else "_test"

        stats = self.get_stats()
        if stats == {}:
            self.logging_struct.py_logger.warning("Stats is empty... are you logging too frequently?")
            return "", {}

        logging_dict =  dict(
                         T_env=self.T_env,
                        )

        logging_dict["episode_reward"+test_suffix] = _seq_mean(stats["episode_reward"+test_suffix])
        logging_dict["episode_length"+test_suffix] = _seq_mean(stats["episode_length"+test_suffix])
        if "policy_entropy"+test_suffix in stats:
            logging_dict["policy_entropy"+test_suffix] = _seq_mean(stats["policy_entropy"+test_suffix])
        if "q_entropy"+test_suffix in stats:
            logging_dict["q_entropy"+test_suffix] = _seq_mean(stats["q_entropy"+test_suffix])

        logging_str = ""
        logging_str += _make_logging_str(_copy_remove_keys(logging_dict, ["T_env"+test_suffix]))

        if self.env_stats_aggregator is not None:
            # get logging str from env_stats aggregator
            logging_str += self.env_stats_aggregator.log(log_directly = False)

        if log_directly:
            self.logging_struct.py_logger.info("{} RUNNER INFO: {}".format("TEST" if self.test_mode else "TRAIN",
                                                                           logging_str))
        return logging_str, logging_dict


    def get_stats(self):
        if hasattr(self, "_stats"):
            tmp = deepcopy(self._stats)
            self._stats={}
            return tmp
        else:
            return []
