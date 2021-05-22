# REFIL
Code for [*Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2006.04222) (Iqbal et al., ICML 2021)

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent reinforcement learning algorithms.

# Dependencies
- Docker
- NVIDIA-Docker (if you want to use GPUs)

## Setup instructions

Build the Dockerfile using 
```shell
cd docker
./build.sh
```

Set up StarCraft II.

```shell
./install_sc2.sh
```

## Run an experiment 

Run an `ALGORITHM` from the folder `src/config/algs`
in an `ENVIRONMENT` from the folder `src/config/envs`
on a specific `GPU` using some `PARAMETERS`:
```shell
./run.sh <GPU> src/main.py --env-config=<ENVIRONMENT> --config=<ALGORITHM> with <PARAMETERS>
```

Possible environments are:
- `group_matching`: Group Matching environment from the paper
- `sc2custom`: StarCraft environment from the paper

For StarCraft you need to specify the set of tasks to train on by including the parameter `scenario=<scenario_set_name>`.
Here are the possible scenario sets:

- Included in the paper:
    - `3-8sz_symmetric`
    - `3-8MMM_symmetric`
    - `3-8csz_symmetric`
- Debugging/Additional:
    - `3-8m_symmetric`
    - `6-11m_mandown`

Possible algorithms are:
- `refil`: REFIL (our method)
- `refil_group_matching`: REFIL w/ hyperparameters for Group Matching game
- `qmix_atten`: QMIX (Attention)
- `qmix_atten_group_matching`: QMIX (Attention) w/ hyperparameters for Group Matching game
- `refil_vdn`: REFIL (VDN)
- `vdn_atten`: VDN (Attention)

For group matching oracle methods, include the following parameters while selecting `refil_group_matching` as the algorithm:
- REFIL (Fixed Oracle): `train_gt_factors=True`
- REFIL (Randomized Oracle): `train_rand_gt_factors=True`

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@InProceedings{iqbal2021refil,
  title={Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning},
  author={Iqbal, Shariq and de Witt, Christian A Schroeder and Peng, Bei and B{\"o}hmer, Wendelin and Whiteson, Shimon and Sha, Fei},
  booktitle =    {Proceedings of the 38th International Conference on Machine Learning},
  year =     {2021},
  series =   {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```
