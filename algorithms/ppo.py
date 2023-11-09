import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from amtl.optim import * # for gradient manipulation

from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 optimiser_encoder_pol=None,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 grad_correction='none',
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        self.policy_separate_gru = self.args.policy_separate_gru
        self.grad_correction = grad_correction


        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)

        if self.grad_correction != 'none':
            if self.grad_correction == 'nash':
                self.balancer = get_method(self.grad_correction, n_tasks = self.args.num_processes)
            else:
                self.balancer = get_method(self.grad_correction)

        self.optimiser_vae = optimiser_vae
        self.optimiser_encoder_pol = optimiser_encoder_pol

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        self.lr_scheduler_encoder_pol = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
            if hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
                self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)
            if hasattr(self.args, 'policy_separate_gru') and self.args.policy_separate_gru:
                self.lr_scheduler_encoder_pol = optim.lr_scheduler.LambdaLR(self.optimiser_encoder_pol, lr_lambda=lam)

    def update(self,
               policy_storage,
               encoder=None,  # VAE
               encoder_pol = None, #policy encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               policy_separate_gru = False,
               compute_vae_loss=None  # function that can compute the VAE loss
               ):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)


        # if this is true, we will update the VAE at every PPO update
        # otherwise, we update it after we update the policy

        # update the normalisation parameters of policy inputs before updating
        #the normalization should be detached, so it appears before recompute_embeddings
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # TODO this does not work for variBAD rllloss through encoder, need to figure out why,
        if rlloss_through_encoder:
            # recompute embeddings (to build computation graph)
            utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=0,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None, mixture=self.args.vae_mixture_num>1,
                                     policy_separate_gru=False)
        elif policy_separate_gru:
            utl.recompute_embeddings(policy_storage, encoder_pol, sample=False, update_idx=0,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                     policy_separate_gru=True)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):
            #print('ppo epoch: ', e)
            if self.grad_correction == 'none':
                data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            else:
                data_generator = policy_storage.feed_forward_generator_uniform(advantages, self.num_mini_batch)

            for sample in data_generator:

                state_batch, belief_batch, task_batch, prob_batch, latent_pol_batch,\
                actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                if not rlloss_through_encoder:
                    state_batch = state_batch.detach()
                    if latent_sample_batch is not None:
                        latent_sample_batch = latent_sample_batch.detach()
                        latent_mean_batch = latent_mean_batch.detach()
                        latent_logvar_batch = latent_logvar_batch.detach()
                    if prob_batch is not None:
                        prob_batch = prob_batch.detach()

                latent_batch = utl.get_latent_for_policy(args=self.args, latent_sample=latent_sample_batch,
                                                         latent_mean=latent_mean_batch,
                                                         latent_logvar=latent_logvar_batch
                                                         )

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy_ew, dist_entropy = \
                    self.actor_critic.evaluate_actions(state=state_batch, latent=latent_batch,
                                                       belief=belief_batch, task=task_batch, prob=prob_batch,latent_pol = latent_pol_batch,
                                                       action=actions_batch, entropy_mean = False)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2)
                action_loss_ew = action_loss.mean() #equal weighting

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)

                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2)
                value_loss_ew = value_loss.mean() #equal weighting

                if self.grad_correction == 'none':
                    # zero out the gradients
                    self.optimiser.zero_grad()
                    if rlloss_through_encoder:
                        self.optimiser_vae.zero_grad()
                    if policy_separate_gru:
                        self.optimiser_encoder_pol.zero_grad()

                    # compute policy loss and backprop
                    loss = value_loss_ew * self.value_loss_coef + action_loss_ew - dist_entropy_ew * self.entropy_coef

                    # compute vae loss and backprop
                    if rlloss_through_encoder:
                        loss += self.args.vae_loss_coeff * compute_vae_loss()

                    # compute gradients (will attach to all networks involved in this computation)
                    loss.backward()

                    # clip gradients
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
                    if rlloss_through_encoder:
                        if self.args.encoder_max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(encoder.parameters(), self.args.encoder_max_grad_norm)
                    if policy_separate_gru:
                        if self.args.encoder_max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(encoder_pol.parameters(), self.args.encoder_max_grad_norm)

                    # update
                    self.optimiser.step()
                    if rlloss_through_encoder:
                        self.optimiser_vae.step()
                    if policy_separate_gru:
                        self.optimiser_encoder_pol.step()

                    loss_epoch += loss.item()

                elif self.args.vae_mixture_num>1 and self.args.task_identification=='argmax':
                    n=self.args.vae_mixture_num
                    subtask_indices = torch.argmax(prob_batch, dim=1)
                    mask = subtask_indices.unsqueeze(-1)==torch.arange(n).to(device)
                    indices_dict =  {i: mask[:, i].nonzero() for i in range(n)}
                    active_indices = []
                    for i in range(n):
                        if len(indices_dict[i]) != 0:  # Check if the element exists in the tensor
                            active_indices.append(i)

                    self.optimiser.zero_grad()
                    losses = {}
                    for ind in active_indices:
                        loss_range = indices_dict[ind]
                        proc_loss = value_loss[loss_range].mean() * self.value_loss_coef + \
                                    action_loss[loss_range].mean() - \
                                    dist_entropy[loss_range].mean() * self.entropy_coef
                        losses[ind] = proc_loss

                    ###RECOMPUTE GRADIENTS
                    self.balancer.step(
                        losses = losses,
                        shared_params = list(self.actor_critic.parameters()),
                        task_specific_params = None
                    )

                    self.optimiser.step()
                    loss_epoch += sum(losses.values()).item()/self.args.num_processes

                else: #pcgrad, cagrad...etc
                    self.optimiser.zero_grad()

                    losses = {}
                    samplers_per_proc = policy_storage.num_steps // self.num_mini_batch
                    for proc in range(self.args.num_processes):
                        loss_range = range(samplers_per_proc*proc,samplers_per_proc*(proc+1))
                        proc_loss = value_loss[loss_range].mean() * self.value_loss_coef + \
                                    action_loss[loss_range].mean() - \
                                    dist_entropy[loss_range].mean() * self.entropy_coef
                        losses[proc] = proc_loss

                    ###RECOMPUTE GRADIENTS
                    self.balancer.step(
                        losses = losses,
                        shared_params = list(self.actor_critic.parameters()),
                        task_specific_params = None
                    )

                    self.optimiser.step()
                    loss_epoch += sum(losses.values()).item()/self.args.num_processes

                value_loss_epoch += value_loss_ew.item()
                action_loss_epoch += action_loss_ew.item()
                dist_entropy_epoch += dist_entropy_ew.item()

                if rlloss_through_encoder:
                    # recompute embeddings (to build computation graph)
                    utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=e + 1,
                                             detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None, mixture=self.args.vae_mixture_num>1,
                                             policy_separate_gru = False)
                elif policy_separate_gru:
                    utl.recompute_embeddings(policy_storage, encoder_pol, sample=False, update_idx=e + 1,
                                             detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                             policy_separate_gru = True)

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(self.args.num_vae_updates):
                compute_vae_loss(update=True)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()
        if self.lr_scheduler_encoder_pol is not None:
            self.lr_scheduler_encoder_pol.step()
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch

    def act(self, state, latent, belief, task, prob, latent_pol, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, prob=prob, latent_pol=latent_pol, deterministic=deterministic)
