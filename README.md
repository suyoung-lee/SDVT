#SDVT

Supplemantary material of Subtask Decomposition and Virtual Training (SDVT) submitted for NeurIPS 2023.

Please do not distribute.


### Requirements
Our code is based on the implementation of [VariBAD](https://github.com/lmzintgraf/varibad) on Python 3.9. 

Please refer to the `requirements.txt`

`pip install -r requirements.txt`

In addition to those requirements, you need to install [metaworld](https://github.com/Farama-Foundation/Metaworld) to run experiments on ML-10 and ML-45.

    #IMPORTANT
    On May 9, 2023, there has been a large update in Meta-World, that includes changes in some tasks.
    It is recommended to use previous version of Meta-World
    to maintain consistency with previous works and to reproduce our results.
    
    git clone https://github.com/Farama-Foundation/Metaworld.git
    cd Metaworld
    pip install -e .
    git reset --hard 04be337a12305e393c0caf0cbf5ec7755c7c8feb


### Overview

The main parts of our implementations are at `metalearner_ml10_...py`, `metalearner_ml45_...py` and the GMVAE with dispersion at `vae_mixture_ext.py`.

### Reproducing the results
All the configuration files are in `config/` folder.

We use 8 seeds from 20 to 27 (written as Seed 0 to Seed 7 in the manuscript for brevity).

* To run our SDVT algorithm on ML-10,

`python main.py --env-type ml10-SDVT`


* You may play around different hyperparameters in the `config/` folder as follows

`python main.py --env-type ml10-SDVT --vae_mixture_num 5 --cat_loss_coeff 0.5`


* To run SDVT without virtual training,

`python main.py --env-type ml10-SD`

* To run the lightweight version of our methods,

`python main.py --env-type ml10-SDVT_LW`

`python main.py --env-type ml10-SD_LW`


* To reproduce the result of variBAD and LDM,

`python main.py --env-type ml10-VariBAD`

`python main.py --env-type ml10-LDM`

* To load and evaluate trained models,

`python main.py --env-type ml10-eval --load-dir ... --load-iter ...`


To run experiments on ML-45, replace the ml10 into ml45 for all commands above. 

All the results will be logged at `logs/` folder.



To reproduce results of RL2, MAML, and PEARL, use the [garage](https://github.com/rlworkgroup/garage/pull/2287) repository.

You have to use the exact pull request used to report the results in the Meta-World paper.


