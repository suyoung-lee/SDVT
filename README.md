#SDVT

Supplemantary material of Subtask Decomposition and Virtual Training (SDVT) for ICML 2023.

Please do not distribute.


### Requirements
Our code is based on the implementation of [VariBAD](https://github.com/lmzintgraf/varibad). Please refer to the `requirements.txt`

In addition to the requirements there, you need to install [metaworld](https://github.com/Farama-Foundation/Metaworld) and cv2.

### Overview

The main part of our implementation is at `metalearner_ml10.py`, the GMVAE with dispersion at `vae_mixture_ext.py`.

There's quite a bit of documentation in the respective scripts so have a look there for details.

### Reproducing the results

To run our SDVT algorithm on ML-10,

`python main.py --env-type gridworld_varibad`
 
It will use the default hyperparmeters at `config/gridworld/args_grid_varibad.py`
and the results will be logged at `logs/` directory.


To run SDVT without virtual training,

`python main.py --env-type gridworld_varibad`

To reproduce the result of variBAD and LDM,

`python main.py --env-type gridworld_varibad`

`python main.py --env-type gridworld_varibad`

To reproduce results of RL2, MAML, and PEARL, use the [garage](https://github.com/rlworkgroup/garage/pull/2287) repository.



