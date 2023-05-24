"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(self,
                 args, num_steps, num_processes,
                 state_dim, belief_dim, task_dim,prob_dim,
                 action_space,
                 hidden_size, latent_dim, normalise_rewards):

        self.args = args
        self.policy_separate_gru = self.args.policy_separate_gru
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim
        self.prob_dim = prob_dim

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, state_dim)
        if not self.args.vae_mixture_num > 1:
            self.y = self.z = self.mu = self.var = self.logits = self.prob = None
        if self.args.pass_latent_to_policy:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            if self.args.vae_mixture_num > 1:
                self.y = []##
                #self.z = []
                #self.mu = []
                #self.var = []
                #self.logits = []
                #self.prob = torch.zeros(num_steps+1, num_processes, prob_dim)
                self.prob = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
            self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)

            # next_state will include s_N when state was reset, skipping s_0
            # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
            self.next_state = torch.zeros(num_steps, num_processes, state_dim)
        else:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_samples = None
            self.latent_y = None ##
            #self.latent_z = None
            #self.mu = None
            #self.var = None
            #self.logits = None
            self.prob = None
        if self.args.pass_belief_to_policy:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        if self.args.pass_task_to_policy:
            self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)
        else:
            self.tasks = None
        if self.policy_separate_gru:
            self.latent_pol = []
            self.hidden_states_pol = torch.zeros(num_steps + 1, num_processes, hidden_size)

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None
        self.action_log_probs_dimwise = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.to_device()

    def to_device(self):
        if self.args.pass_state_to_policy:
            self.prev_state = self.prev_state.to(device)
        if self.args.pass_latent_to_policy:
            self.latent_samples = [t.to(device) for t in self.latent_samples]
            self.latent_mean = [t.to(device) for t in self.latent_mean]
            self.latent_logvar = [t.to(device) for t in self.latent_logvar]
            if self.args.vae_mixture_num > 1:
                self.y = [t.to(device) for t in self.y] ##
                #self.z = [t.to(device) for t in self.z] ##
                #self.mu = [t.to(device) for t in self.mu] ##
                #self.var = [t.to(device) for t in self.var] ##
                #self.logits = [t.to(device) for t in self.logits] ##
                self.prob = [t.to(device) for t in self.prob] ##
            self.hidden_states = self.hidden_states.to(device)
            self.next_state = self.next_state.to(device)
        if self.policy_separate_gru:
            self.latent_pol = [t.to(device) for t in self.latent_pol]
            self.hidden_states_pol = self.hidden_states_pol.to(device)


        if self.args.pass_belief_to_policy:
            self.beliefs = self.beliefs.to(device)
        if self.args.pass_task_to_policy:
            self.tasks = self.tasks.to(device)
        self.rewards_raw = self.rewards_raw.to(device)
        self.rewards_normalised = self.rewards_normalised.to(device)
        self.done = self.done.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)

    def insert(self,
               state,
               belief,
               task,
               actions,
               rewards_raw,
               rewards_normalised,
               value_preds,
               masks,
               bad_masks,
               done,
               #
               hidden_states=None,
               latent_sample=None,
               latent_mean=None,
               latent_logvar=None,
               y=None,
               #z=None,
               #mu=None,
               #var=None,
               #logits=None,
               prob=None,
               latent_pol=None,
               hidden_states_pol=None
               ):
        self.prev_state[self.step + 1].copy_(state)
        if self.args.pass_belief_to_policy:
            self.beliefs[self.step + 1].copy_(belief)
        if self.args.pass_task_to_policy:
            self.tasks[self.step + 1].copy_(task)
        if self.args.pass_latent_to_policy:
            self.latent_samples.append(latent_sample.detach().clone())
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
            self.hidden_states[self.step + 1].copy_(hidden_states.detach())
            if self.args.vae_mixture_num>1:
                self.y.append(y.detach().clone())#
                #self.z.append(z.detach().clone())#
                #self.mu.append(mu.detach().clone())#
                #self.var.append(var.detach().clone())#
                #self.logits.append(logits.detach().clone())#
                #self.prob[self.step + 1].copy_(prob.detach().clone())
                self.prob.append(prob.detach().clone())

        if self.policy_separate_gru:
            self.latent_pol.append(latent_pol.detach().clone())
            self.hidden_states_pol[self.step + 1].copy_(hidden_states_pol.detach())


        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.pass_belief_to_policy:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.args.pass_task_to_policy:
            self.tasks[0].copy_(self.tasks[-1])
        if self.args.pass_latent_to_policy:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            self.hidden_states[0].copy_(self.hidden_states[-1])
            if self.args.vae_mixture_num > 1:
                self.y = []
                #self.z = []
                #self.mu = []
                #self.var = []
                #self.logits = []
                #self.prob[0].copy_(self.prob[-1])
                self.prob=[]
        if self.policy_separate_gru:
            self.latent_pol = []
            self.hidden_states_pol[0].copy_(self.hidden_states_pol[-1])

        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.action_log_probs = None
        self.action_log_probs_dimwise = None

    def compute_returns(self, next_value, use_gae, gamma, tau, use_proper_time_limits=True):

        if self.normalise_rewards:
            rewards = self.rewards_normalised.clone()
        else:
            rewards = self.rewards_raw.clone()

        self._compute_returns(next_value=next_value, rewards=rewards, value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=gamma, tau=tau, use_gae=use_gae, use_proper_time_limits=use_proper_time_limits)

    def _compute_returns(self, next_value, rewards, value_preds, returns, gamma, tau, use_gae, use_proper_time_limits):

        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        return len(self.prev_state) * self.num_processes

    def before_update(self, policy):
        latent = utl.get_latent_for_policy(self.args,
                                           latent_sample=torch.stack(
                                               self.latent_samples[:-1]) if self.latent_samples is not None else None,
                                           latent_mean=torch.stack(
                                               self.latent_mean[:-1]) if self.latent_mean is not None else None,
                                           latent_logvar=torch.stack(
                                               self.latent_logvar[:-1]) if self.latent_mean is not None else None)

        if self.prob is not None:
            prob = torch.stack(self.prob[:-1])
        else:
            prob = None

        if self.policy_separate_gru:
            latent_pol = torch.stack(self.latent_pol[:-1]) if self.latent_pol is not None else None
        else:
            latent_pol = None

        _, action_log_probs, _ = policy.evaluate_actions(self.prev_state[:-1],
                                                         latent,
                                                         self.beliefs[:-1] if self.beliefs is not None else None,
                                                         self.tasks[:-1] if self.tasks is not None else None,
                                                         prob,
                                                         latent_pol,
                                                         self.actions)
        self.action_log_probs = action_log_probs.detach() #[500, 10, 1]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.args.pass_state_to_policy:
                state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            else:
                state_batch = None
            if self.args.pass_latent_to_policy:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None
            if self.args.pass_prob_to_policy:
                prob_batch = torch.cat(self.prob[:-1])[indices]
            else:
                prob_batch = None
            if self.args.pass_belief_to_policy:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            if self.args.pass_task_to_policy:
                task_batch = self.tasks[:-1].reshape(-1, *self.tasks.size()[2:])[indices]
            else:
                task_batch = None
            if self.policy_separate_gru:
                latent_pol_batch = torch.cat(self.latent_pol[:-1])[indices]
            else:
                latent_pol_batch = None

            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, belief_batch, task_batch, prob_batch, latent_pol_batch,\
                  actions_batch, \
                  latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
