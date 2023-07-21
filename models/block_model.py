import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

### Add block model components
from models.attention_model import BaseModel


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Block_model(nn.Module):
    def __init__(self, input_dim, top_k, block_hid_dim, att_layer_num, norm_type, device, bl_log_sig_min,
                 bl_log_sig_max):
        super().__init__()

        self.input_dim = input_dim #64 (hs+hr+ha)

        ### Block latent variable(embedding) dimension
        self.block_hid_dim = block_hid_dim  # 256
        self.block_mu_size = 64

        ### Attention selection
        self.top_k = top_k

        self.device = device
        self.bl_log_sig_min = bl_log_sig_min
        self.bl_log_sig_max = bl_log_sig_max

        activation = nn.Tanh()
        # activation = nn.ReLU()

        # Define block memory
        ## Blockwise RNN. For simplicity, we use GRU. However, LSTM can also be applied.
        self.block_memory_rnn = nn.GRUCell(self.input_dim * self.top_k, self.block_hid_dim)

        ## Define Self-attention
        ## Note that the output dimension of self-attention is the same as input dimension due to Residual connection.
        ## 4 multi-heads
        self.att_model_q = BaseModel(hidden_dim=self.input_dim, num_layers=att_layer_num,
                                     norm_type=norm_type)

        ## Define mean of encoder
        self.block_mu = nn.Sequential(
            nn.Linear(self.block_hid_dim, self.block_hid_dim // 2),
            activation,
            nn.Linear(self.block_hid_dim // 2, self.block_mu_size)
        )

        ## Define stddev of decoder
        self.block_sig = nn.Sequential(
            nn.Linear(self.block_hid_dim, self.block_hid_dim // 2),
            activation,
            nn.Linear(self.block_hid_dim // 2, self.block_mu_size),
            # nn.Softplus()
        )

        ## Define p_theta model for self-normalized importance sampling
        self.self_norm_model = nn.Sequential(
            nn.Linear(self.input_dim * self.top_k + self.block_mu_size, self.input_dim),
            activation,
            nn.Linear(self.input_dim, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs_block_ori, block_memory_ori):
        ## obs_block_ori: [ seq=L, batch=16, 64(feature)]
        ## block_memory_ori: [16, self.block_hid_dim=256]

        obs_block = obs_block_ori
        # print("obs_block", obs_block.shape)
        assert len(obs_block.size()) == 3

        block_memory = block_memory_ori
        # print("block_memory", block_memory.shape)
        assert len(block_memory.size()) == 2

        # print("block_memory first", block_memory_ori)

        # Step 1. input obs and action to Attention for model_q.
        # Then select some of them. Default is to choose 2 ends like bi-LSTM (block_len >=2).
        # Later, we are going to choose 'self.trans_select_N' number of outputs based on Attention score.

        trans_q_output, attention_matrix = self.att_model_q.forward(obs_block)  # (seq, batch_size=4, 64)
        trans_q_output = trans_q_output.permute(1, 0, 2)  # (batch=4, seq, 256)
        # print("trans_q_output", trans_q_output, trans_q_output.shape) # (batch, seq, hidden_dim=64)
        # print("attention", attention_matrix, attention_matrix.shape) # (batch*n_head, trans_seq, trans_seq)

        attention_matrix_align = torch.cat(attention_matrix.split(obs_block.size()[1], dim=0),
                                           2)  # (batch, trans_seq, trans_seq*n_head)

        ### Pass top K elements
        batch_here = trans_q_output.shape[0]
        seq_len_here = trans_q_output.shape[1]

        _, top_k_index = torch.topk(attention_matrix_align.sum(dim=-1), k=min(self.top_k, seq_len_here), dim=-1)

        top_k_index_repeat = top_k_index.unsqueeze(dim=-1).repeat(1, 1, trans_q_output.shape[2])
        # (batch, min(self.top_k, trans_q_output.shape[1]), 256)
        top_k_q_output_selected = torch.gather(trans_q_output, dim=1, index=top_k_index_repeat)  # selected vectors
        # (batch, min(self.top_k, trans_q_output.shape[1]), 256)

        reshaped = torch.reshape(top_k_q_output_selected,
                                 (batch_here, -1))  # (batch, min(self.top_k, trans_q_output.shape[1])*256)
        # print("reshaped init", reshaped, reshaped.shape) Y_ns

        ## Padding if necessary
        if seq_len_here < self.top_k:
            zero_pad = torch.zeros((batch_here, (self.top_k - seq_len_here) * trans_q_output.shape[2]),
                                   device=self.device)
            reshaped = torch.cat((reshaped, zero_pad), dim=-1)  # (batch, self.top_k*256)

        # Step 2. Here, 'block_memory' should be changed to block_variable recurrently.
        output = self.block_memory_rnn(reshaped, block_memory)  # (batch, hidden=256)

        return output
        #block_memory_ori = block_memory
        #return block_memory_ori, reshaped



    def block_mu_sig(self, block_memory):
        sig = torch.exp(self.block_sig(block_memory).clamp(self.bl_log_sig_min, self.bl_log_sig_max))

        return torch.cat((self.block_mu(block_memory), sig), dim=-1)