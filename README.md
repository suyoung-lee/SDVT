# Solving ML10

CUDA_VISIBLE_DEVICES=4 python main.py --env-type ml10 --seed 10 --vae_mixture_num 5 --latent_dim 5 --vae_avg_elbo_terms True --vae_avg_reconstruction_terms True  --rew_loss_coeff 10 --state_loss_coeff 10000 --pass_prob_to_policy True --results_log_dir ./logs/vae10x_c5_d5_passprob_avgvae_adjustalpha

CUDA_VISIBLE_DEVICES=0 python main.py --env-type ml10-eval --load-dir logs/vae_10x/logs_ML10Env-v2/varibad_12__14:09_10:49:58/ --load-iter 59999 --render True

CUDA_VISIBLE_DEVICES=0 python main.py --env-type ml10-eval --load-dir ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/varibad_12__14:09_10:49:58/ --load-iter 29999

CUDA_VISIBLE_DEVICES=0 python main.py --env-type ml10 --load-dir ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/varibad_12__14:09_10:49:58/ --load-iter 29999

CUDA_VISIBLE_DEVICES=0 python main.py --env-type ml10-post --load-dir ml10-joint-log/vae10x_c5_d5_passprob_avgvae_adjustalpha/logs_ML10Env-v2/varibad_12__14:09_10:49:58/ --load-iter 29999
