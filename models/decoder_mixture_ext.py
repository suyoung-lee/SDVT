#v8 fixed
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StateTransitionDecoder_mixture_ext(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 class_dim,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 pred_type='deterministic',
                 dropout_rate=0.0
                 ):
        super(StateTransitionDecoder_mixture_ext, self).__init__()

        self.args = args

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.drop_input = dropout_rate>0.0

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)
        if self.drop_input:
            ha = self.dropout(ha)
            hs = self.dropout(hs)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder_mixture_ext(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 class_dim,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 dropout_rate= 0.0
                 ):
        super(RewardDecoder_mixture_ext, self).__init__()

        self.args = args

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action
        self.dropout = nn.Dropout(p=dropout_rate)
        self.drop_input = dropout_rate>0.0

        if self.multi_head:
            # one output head per state to predict rewards
            curr_input_dim = latent_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]
            self.fc_out = nn.Linear(curr_input_dim, num_states)
        else:
            # get state as input and predict reward prob
            self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
            if self.input_action:
                self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
            else:
                self.action_encoder = None
            curr_input_dim = self.args.encoder_gru_hidden_size + state_embed_dim #because of the dispersion structure
            if input_prev_state:
                curr_input_dim += state_embed_dim
            if input_action:
                curr_input_dim += action_embed_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]

            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(curr_input_dim, 2)
            else:
                self.fc_out = nn.Linear(curr_input_dim, 1)

        self.y_mu = nn.Linear(class_dim, latent_dim)
        self.y_var = nn.Linear(class_dim, latent_dim)

        self.ext1 = nn.Linear(latent_dim, 256)
        self.ext2 = nn.Linear(256, 256)
        self.ext3 = nn.Linear(256, self.args.encoder_gru_hidden_size)

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    def forward(self, latent_state, next_state, prev_state=None, actions=None, y=None):

        # we do the action-normalisation (the the env bounds) here
        if actions is not None:
            actions = utl.squash_action(actions, self.args)

        if self.multi_head:
            h = latent_state.clone()
        else:
            h = F.relu(self.ext1(latent_state))
            h = F.relu(self.ext2(h))
            h_hat = self.ext3(h)

            hns = self.state_encoder(next_state)
            if self.drop_input:
                hns = self.dropout(hns)

            h = torch.cat((h_hat, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                if self.drop_input:
                    ha = self.dropout(ha)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                if self.drop_input:
                    hps = self.dropout(hps)
                h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        if y is None:
            return self.fc_out(h)
        else:
            y_mu, y_var = self.pzy(y)
            return self.fc_out(h), y_mu, y_var, h_hat


class TaskDecoder_mixture_ext(nn.Module):
    def __init__(self,
                 layers,
                 class_dim,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 ):
        super(TaskDecoder_mixture_ext, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):

        h = latent_state

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)
