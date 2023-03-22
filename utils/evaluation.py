import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder=None,
             num_episodes=None
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name,
                         seed=args.seed * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         add_done_info=args.max_rollouts_per_task > 1,
                         )
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task = utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              deterministic=False
                                              )

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            done_mdp = [info['done_mdp'] for info in infos]

            if encoder is not None:
                # update the hidden state
                if args.vae_mixture_num>1:
                    latent_sample, latent_mean, latent_logvar, hidden_state, \
                    y, z, mu, var, logits, prob  = utl.update_encoding(encoder=encoder,
                                                                                                  next_obs=state,
                                                                                                  action=action,
                                                                                                  reward=rew_raw,
                                                                                                  done=None,
                                                                                                  hidden_state=hidden_state,
                                                                                                  vae_mixture_num = args.vae_mixture_num)
                else:
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                                  next_obs=state,
                                                                                                  action=action,
                                                                                                  reward=rew_raw,
                                                                                                  done=None,
                                                                                                  hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()

    return returns_per_episode[:, :num_episodes]

def evaluate_metaworld(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder=None,
             num_episodes=None,
             test = False, task_list = None, save_episode_probs=False, save_episode_successes=False):
    env_name = args.env_name
    if test:
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = 10

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---
    envs = make_vec_envs(env_name,
                         seed=args.seed,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         add_done_info=args.max_rollouts_per_task > 1,
                         )
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task = utl.reset_env(envs, args)
    envs.reset_task(task_list)
    state2= torch.from_numpy(envs._get_obs()).float().to(device)
    state[:,:39] = state2.clone()

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)
    prob = None
    episode_probs = None
    episode_successes = None
    if save_episode_successes:
        episode_successes = np.zeros((num_processes, num_episodes))

    if encoder is not None:
        # reset latent state to prior
        if args.vae_mixture_num>1:
            prior_y, prior_z, prior_mu, prior_var, prior_logits, prior_prob, prior_hidden_state = encoder.prior_mixture(num_processes)
            latent_sample = prior_z
            latent_mean = prior_mu
            latent_logvar = torch.log(prior_var + 1e-20)
            hidden_state = prior_hidden_state
            prob = prior_prob
            if save_episode_probs:
                episode_probs = torch.zeros((num_processes, num_episodes + 1, num_steps, args.vae_mixture_num)).to(device)

        else:
            latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)

    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    meta_successes = np.zeros(num_processes)
    for episode_idx in range(num_episodes):
        successes = np.zeros(num_processes)
        if args.render: #save videos for each rollout episode
            for i in range(num_processes):
                os.makedirs(
                    'renders/' + args.load_dir + '{}/task{:2d}/subtask{:2d}'.format(args.load_iter, task_list[i, 0],
                                                                                    task_list[i, 1]), exist_ok=True)
            imgs_array = []
        for step_idx in range(num_steps):
            with torch.no_grad():
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=state,
                                              belief=belief,
                                              task=task,
                                              prob=prob,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              #deterministic=True,
                                              deterministic=False
                                              )

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)

            done_mdp = [info['done_mdp'] for info in infos]
            successes = np.fmax(successes, [info['success'] for info in infos])

            if args.render:
                imgs = np.array([info['image'] for info in infos])
                success_list = [info['success'] for info in infos]
                for i in range(num_processes):
                    imgs[i,0:30,0:30,:] = int(255*success_list[i]) #to distinguish success
                imgs_array.append(imgs)

            if encoder is not None:
                # update the hidden state
                if args.vae_mixture_num>1:
                    latent_sample, latent_mean, latent_logvar, hidden_state,\
                    y, z, mu, var, logits, prob = utl.update_encoding(encoder=encoder,
                                                                                                  next_obs=state,
                                                                                                  action=action,
                                                                                                  reward=rew_raw,
                                                                                                  done=None,
                                                                                                  hidden_state=hidden_state,
                                                                      vae_mixture_num=args.vae_mixture_num)
                    if save_episode_probs:
                        episode_probs[:, episode_idx, step_idx , :] = prob.clone()

                else:
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                                  next_obs=state,
                                                                                                  action=action,
                                                                                                  reward=rew_raw,
                                                                                                  done=None,
                                                                                                  hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

        if args.render:
            imgs_array = np.array(imgs_array)
            for i in range(num_processes):
                img_array = imgs_array[:,i,:,:,:] #(500x10x480x640x3) to (500x480x640x3)
                pathout = 'renders/'+args.load_dir+ '{}/task{:2d}/subtask{:2d}/task{:2d}_subtask{:2d}_epi_{:2d}.mp4'.format(args.load_iter,task_list[i,0],task_list[i,1],task_list[i,0],task_list[i,1],episode_idx)
                out = cv2.VideoWriter(pathout, cv2.VideoWriter_fourcc(*'mp4v'), 50, (np.shape(img_array)[2],np.shape(img_array)[1])) #width, height
                horizon = np.shape(img_array)[0]
                for j in range(horizon):
                    rgb_img = cv2.cvtColor(img_array[j], cv2.COLOR_RGB2BGR)
                    out.write(rgb_img)
                out.release()

        if save_episode_successes:
            episode_successes[:, episode_idx] = successes
        meta_successes = meta_successes+successes
    envs.close()
    meta_successes = meta_successes/num_episodes

    if args.vae_mixture_num > 1:
        prob =prob.detach().cpu().numpy()
        if save_episode_probs:
            episode_probs = episode_probs[:, :num_episodes, :, :].detach().cpu().numpy()
    return returns_per_episode[:, :num_episodes].detach().cpu().numpy(), latent_mean.detach().cpu().numpy(), \
                                                         latent_logvar.detach().cpu().numpy(), meta_successes, prob, episode_probs, episode_successes




