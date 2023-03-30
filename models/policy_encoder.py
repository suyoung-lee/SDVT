import warnings

import gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder, RNNEncoder_keepdim
from utils.helpers import get_task_dim, get_num_tasks
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyEncoder:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(self.args) if self.args.decode_task else None
        self.num_tasks = get_num_tasks(self.args) if self.args.decode_task else None

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        self.optimiser_vae = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr_vae)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder without keeping the output dimension"""
        encoder = RNNEncoder_keepdim(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            #latent_dim=self.args.encoder_gru_hidden_size, #do not reduce the dimension
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(device)
        return encoder
