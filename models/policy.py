"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from utils import helpers as utl
try:
    from torch.distributions import TanhTransform, TransformedDistribution

    class TanhNormal(TransformedDistribution):
        def __init__(self, base_distribution, transforms, validate_args=None):
            super().__init__(base_distribution, transforms, validate_args=None)

    @property
    def mean(self):
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

except ImportError:
    print('You are probably running MuJoCo 131, so PyTorch Transforms cannot be used. '
          'Do not set norm_actions_pre_sampling, this will break.')
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_latent_to_policy,
                 pass_belief_to_policy,
                 pass_task_to_policy,
                 pass_prob_to_policy,
                 dim_state,
                 dim_latent,
                 dim_belief,
                 dim_task,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 policy_initialisation,  # orthogonal / normc
                 # output
                 action_space,
                 init_std,
                 min_std,
                 max_std,
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()

        self.args = args

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        if policy_initialisation == 'normc':
            init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))
        elif policy_initialisation == 'orthogonal':
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_prob_to_policy = pass_prob_to_policy
        self.pass_task_to_policy = pass_task_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy
        self.policy_separate_gru = self.args.policy_separate_gru
        self.dim_latent_pol  = self.args.encoder_gru_hidden_size


        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = self.args.norm_state_for_policy and (dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(shape=(dim_state)) #for some ML-45 diverging tasks
        self.norm_latent = self.args.norm_latent_for_policy and (dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(shape=(dim_latent))
        self.norm_belief = self.args.norm_belief_for_policy and (dim_belief is not None)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief))
        self.norm_task = self.args.norm_task_for_policy and (dim_task is not None)
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = utl.RunningMeanStd(shape=(dim_task))
        dim_prob = self.args.vae_mixture_num
        self.norm_prob = self.args.norm_prob_for_policy and (dim_prob is not None)
        if self.pass_prob_to_policy and self.norm_prob:
            self.prob_rms = utl.RunningMeanStd(shape=(dim_prob))
        if self.policy_separate_gru and self.norm_latent:
            self.latent_pol_rms = utl.RunningMeanStd(shape=(self.dim_latent_pol))


        curr_input_dim = dim_state * int(self.pass_state_to_policy) + \
                         dim_latent * int(self.pass_latent_to_policy) + \
                         dim_prob * int(self.pass_prob_to_policy) + \
                         dim_belief * int(self.pass_belief_to_policy) + \
                         dim_task * int(self.pass_task_to_policy) + \
                         self.dim_latent_pol * int(self.policy_separate_gru)
        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy_state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder = utl.FeatureExtractor(dim_state, self.args.policy_state_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim
        self.use_latent_encoder = self.args.policy_latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = utl.FeatureExtractor(dim_latent, self.args.policy_latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_latent + self.args.policy_latent_embedding_dim
        self.use_prob_encoder = self.args.policy_prob_embedding_dim is not None
        if self.pass_prob_to_policy and self.use_prob_encoder:
            self.prob_encoder = utl.FeatureExtractor(dim_prob, self.args.policy_prob_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_prob + self.args.policy_prob_embedding_dim
        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = utl.FeatureExtractor(dim_belief, self.args.policy_belief_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy_belief_embedding_dim
        self.use_task_encoder = self.args.policy_task_embedding_dim is not None
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder = utl.FeatureExtractor(dim_task, self.args.policy_task_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_task + self.args.policy_task_embedding_dim
        if self.args.policy_separate_gru and self.use_latent_encoder:
            self.latent_pol_encoder = utl.FeatureExtractor(self.dim_latent_pol, self.args.policy_latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - self.dim_latent_pol + self.args.policy_latent_embedding_dim


        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.actor_layers.append(fc)
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]
        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

        # output distributions of the policy
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_layers[-1], num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, min_std, max_std, self.args.norm_actions_pre_sampling)
        else:
            raise NotImplementedError

    def get_actor_params(self):
        return [*self.actor.parameters(), *self.dist.parameters()]

    def get_critic_params(self):
        return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, state, latent, belief, task, prob, latent_pol):

        # handle inputs (normalise + embed)

        state0= state
        latent0 = latent
        belief0 = belief
        task0 = task

        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            if self.use_state_encoder:
                state = self.state_encoder(state)
        else:
            state = torch.zeros(0, ).to(device)
        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
        else:
            latent = torch.zeros(0, ).to(device)
        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief.float())
        else:
            belief = torch.zeros(0, ).to(device)
        if self.pass_task_to_policy:
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            if self.use_task_encoder:
                task = self.task_encoder(task.float())
        else:
            task = torch.zeros(0, ).to(device)
        if self.pass_prob_to_policy:
            if self.norm_prob:
                prob = (prob - self.prob_rms.mean) / torch.sqrt(self.prob_rms.var + 1e-8)
            if self.use_prob_encoder:
                prob = self.prob_encoder(prob)
        else:
            prob = torch.zeros(0, ).to(device)
        if self.policy_separate_gru:
            if self.norm_latent:
                latent_pol = (latent_pol - self.latent_pol_rms.mean) / torch.sqrt(self.latent_pol_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent_pol = self.latent_pol_encoder(latent_pol)
        else:
            latent_pol = torch.zeros(0, ).to(device)
        # concatenate inputs
        inputs = torch.cat((state, latent, belief, task, prob, latent_pol), dim=-1)


        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state, latent, belief, task, prob=None, latent_pol=None, deterministic=False):
        """
        Returns the (raw) actions and their value.
        """
        value, actor_features = self.forward(state=state, latent=latent, belief=belief, task=task, prob=prob, latent_pol=latent_pol)
        dist = self.dist(actor_features)
        if deterministic:
            if isinstance(dist, FixedCategorical):
                action = dist.mode()
            else:
                action = dist.mean
        else:
            action = dist.sample()

        return value, action

    def get_value(self, state, latent, belief, task, prob, latent_pol=None):
        value, _ = self.forward(state, latent, belief, task, prob, latent_pol)
        return value

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms.update(policy_storage.prev_state[:-1])
        if self.pass_latent_to_policy and self.norm_latent:
            latent = utl.get_latent_for_policy(args,
                                               torch.cat(policy_storage.latent_samples[:-1]),
                                               torch.cat(policy_storage.latent_mean[:-1]),
                                               torch.cat(policy_storage.latent_logvar[:-1])
                                               )
            self.latent_rms.update(latent)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms.update(policy_storage.tasks[:-1])
        if self.pass_prob_to_policy and self.norm_prob:
            self.prob_rms.update(torch.cat(policy_storage.prob[:-1]))
        if self.policy_separate_gru and self.norm_latent:
            self.latent_pol_rms.update(torch.cat(policy_storage.latent_pol[:-1]))


    def evaluate_actions(self, state, latent, belief, task, prob, latent_pol, action, entropy_mean = True):

        value, actor_features = self.forward(state, latent, belief, task, prob, latent_pol)
        dist = self.dist(actor_features)
        if self.args.norm_actions_post_sampling:
            transformation = TanhTransform(cache_size=1)
            dist = TanhNormal(dist, transformation)
            action = transformation(action)
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            # empirical entropy
            # dist_entropy = -action_log_probs.mean()
            # entropy of underlying dist (isn't correct but works well in practice)
            dist_entropy = dist.base_dist.entropy()
        else:
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy()

        if entropy_mean:
            return value, action_log_probs, dist_entropy.mean()
        else:
            return value, action_log_probs, dist_entropy.mean(), dist_entropy


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, min_std, max_std, norm_actions_pre_sampling):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std))
        self.norm_actions_pre_sampling = norm_actions_pre_sampling
        if max_std == 0:
            self.min_std = torch.tensor([1e-6]).to(device)
            self.clip_std = False
        else:
            self.clip_std = True
            self.log_min_std = np.log(min_std)
            self.log_max_std = np.log(max_std)

    def _softclip(self, x, x_min, x_max, alpha=1):

        y_scale = (x_max - x_min) / 2
        y_offset = (x_max + x_min) / 2
        x_scale = (2 * alpha) / (x_max - x_min)
        x_offset = (x_max + x_min) / 2
        return (torch.tanh((x - x_offset) * x_scale) * y_scale) + y_offset

    def forward(self, x):

        action_mean = self.fc_mean(x)
        if self.norm_actions_pre_sampling:
            action_mean = torch.tanh(action_mean)
        if self.clip_std:
            logstd = self._softclip(self.logstd, self.log_min_std, self.log_max_std)
            std = logstd.exp()
        else:
            std = torch.max(self.min_std, self.logstd.exp())

        dist = FixedNormal(action_mean, std)

        return dist


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
