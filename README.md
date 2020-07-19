# AI-QMIX
Code for [*AI-QMIX: Attention and Imagination for Dynamic Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2006.04222) (Iqbal et al., arXiv 2006.04222)

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent reinforcement learning algorithms.

# Dependencies
- Docker
- NVIDIA-Docker (if you want to use GPUs)

## Setup instructions

Build the Dockerfile using 
```
cd docker
./build.sh
```

Set up StarCraft II.

```
./install_sc2.sh
```

## Run an experiment 

Run an ALGORITHM from the folder `src/config/algs`
in an ENVIRONMENT from the folder `src/config/envs`
on a specific GPU using some PARAMETERS:
```
./run.sh <GPU> python3 src/main.py --env-config=<ENVIRONMENT> --config=<ALGORITHM> with <PARAMETERS>
```

Add the `--no-mongo` flag before the "with" statement if you do not have a MongoDB database set up to log results (the results will still be logged in a local file).

Possible environments are:
- `firefighters`: SaveTheCity environment from the paper
- `sc2custom`: StarCraft environment from the paper

For each environment you can specify the set of scenarios to train/test on by including the parameter `with scenario=<scenario_set_name>`.
Here are the possible scenario sets for each environment:
- SaveTheCity
    - '2-8a_2-8b_05': train on 5% of the possible scenarios
    - '2-8a_2-8b_25': train on 25% of the possible scenarios
    - '2-8a_2-8b_45': train on 45% of the possible scenarios
    - '2-8a_2-8b_65': train on 65% of the possible scenarios
    - '2-8a_2-8b_85': train on 85% of the possible scenarios
    - '2-8a_2-8b_sim_Q1': train on the quartile of scenarios least similar to the testing ones
    - '2-8a_2-8b_sim_Q2': train on the quartile of scenarios 2nd least similar to the testing ones
    - '2-8a_2-8b_sim_Q3': train on the quartile of scenarios 2nd most similar to the testing ones
    - '2-8a_2-8b_sim_Q4': train on the quartile of scenarios most similar to the testing ones
- StarCraft
    - '3-8sz_symmetric'
    - '3-8MMM_symmetric'

The testing scenarios are the same on all SaveTheCity scenario set

Possible algorithms are:
- 'imagine_qmix': AI-QMIX (our method)
- 'atten_qmix': A-QMIX
- 'imagine_vdn': AI-VDN
- 'atten_vdn': A-VDN

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```
@article{iqbal2020ai,
  title={AI-QMIX: Attention and Imagination for Dynamic Multi-Agent Reinforcement Learning},
  author={Iqbal, Shariq and de Witt, Christian A Schroeder and Peng, Bei and B{\"o}hmer, Wendelin and Whiteson, Shimon and Sha, Fei},
  journal={arXiv preprint arXiv:2006.04222},
  year={2020}
}
```



