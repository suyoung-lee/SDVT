import math

import torch
import torch.nn as nn


def dot_scaled_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask=True
):
    """ Dot scaled attention
    Implement dot-product scaled attention which takes query, key, value and gives attention scores.

    Arguments:
    query -- Query tensor
                in shape (sequence_length, batch_size, d_k)
    key -- Key tensor
                in shape (sequence_length, batch_size, d_k)
    value -- Value tensor
                in shape (sequence_length, batch_size, d_k)
    padding_mask -- Padding mask tensor in torch.bool type
                in shape (batch_size, sequence_length)
                True for <PAD>, False for non-<PAD>

    Returns:
    attention -- Attention result tensor
                in shape (sequence_length, batch_size, d_k)
    """

    assert query.shape == key.shape == value.shape
    seq_len, _, d_k = query.shape

    QK_t_scaled = torch.bmm(key.permute(1, 0, 2), query.permute(1, 2, 0)) / math.sqrt(d_k)
    # shape: (batch_size, sequence_length, sequence_length)

    print('attention w/o padding',  QK_t_scaled)
    if padding_mask:
        # Create causal mask to mask future timesteps
        future_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().to(device=QK_t_scaled.device)
        future_mask = future_mask[None, :, :]  # add batch dimension

        # Apply the mask
        QK_t_scaled = QK_t_scaled.masked_fill(future_mask, float('-inf'))
        print('attention w/ padding', QK_t_scaled)

    distribution = nn.functional.softmax(QK_t_scaled, dim=2)  # shape: (batch_size, sequence_length, sequence_length)
    print('distribution', distribution)

    attention = torch.bmm(value.permute(1, 2, 0), distribution).permute(2, 0, 1)
    # shape: (sequence_length, batch_size, d_k)

    return attention, distribution


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_head: int = 4
                 ):
        """ Multi-head attention initializer
        Use below attributes to implement the forward function

        Attributes:
        n_head -- the number of heads
        d_k -- Hidden dimension of the dot scaled attention
        V_linear -- Linear function to project hidden_dim of value to d_k
        K_linear -- Linear function to project hidden_dim of key to d_k
        Q_linear -- Linear function to project hidden_dim of query to d_k
        O_linear -- Linear function to project collections of d_k to hidden_dim
        """
        super().__init__()
        assert hidden_dim % n_head == 0
        self.n_head = n_head
        self.d_k = hidden_dim // n_head

        self.V_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.K_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.Q_linear = nn.Linear(hidden_dim, self.n_head * self.d_k, bias=False)
        self.O_linear = nn.Linear(self.n_head * self.d_k, hidden_dim, bias=False)

    def forward(self,
                value: torch.Tensor,
                key: torch.Tensor,
                query: torch.Tensor
                ):
        """ Multi-head attention forward function
        Implement multi-head attention which takes value, key, query, and gives attention score.
        Use dot-scaled attention you have implemented above.

        Note: If you adjust the dimension of batch_size dynamically,
              you can implement this function without any iteration.

        Parameters:
        value -- Value tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        key -- Key tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        query -- Query tensor
                    in shape (sequence_length, batch_size, hidden_dim)

        Returns:
        attention -- Attention result tensor
                    in shape (sequence_length, batch_size, hidden_dim)
        """
        assert value.shape == key.shape == query.shape
        input_shape = value.shape
        seq_length, batch_size, hidden_dim = input_shape

        # permute function does not change the original tensor, just return permuted one.
        # We split and stack different heads with the size: batch_size * self.n_head, seq_len, d_k -> then permute.

        Q_embed_concat = torch.cat(self.Q_linear(query.permute(1, 0, 2)).split(self.d_k, dim=2), 0).permute(1, 0, 2)
        K_embed_concat = torch.cat(self.K_linear(key.permute(1, 0, 2)).split(self.d_k, dim=2), 0).permute(1, 0, 2)
        V_embed_concat = torch.cat(self.V_linear(value.permute(1, 0, 2)).split(self.d_k, dim=2), 0).permute(1, 0, 2)

        attention_stacked, distribution = dot_scaled_attention(query=Q_embed_concat, key=K_embed_concat,
                                                               value=V_embed_concat)

        attention = self.O_linear(torch.cat(attention_stacked.permute(1, 0, 2).split(batch_size, dim=0), 2)).permute(1,
                                                                                                                     0,
                                                                                                                     2)

        assert attention.shape == input_shape
        return attention, distribution


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,  ## same as the number of entities (CO2, etc)
                 dropout: float = .1,
                 n_head: int = 4,
                 norm_type='post'
                 ):
        """ Transformer Encoder Block initializer
        Use below attributes to implement the forward function

        Attributes:
        attention -- Multi-head attention layer
        output -- Output layer
        dropout1, dropout2 -- Dropout layers
        norm1, norm2 -- Layer normalization layers
        """
        super().__init__()

        # Attention Layer
        self.attention = MultiHeadAttention(hidden_dim, n_head)

        # Output Layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
            nn.GELU(),
            # nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        )

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Layer Normalization Layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.norm_type = norm_type

    def forward(self, x: torch.Tensor):

        input_shape = x.shape

        if self.norm_type == 'pre':
            ## PreLN code
            layernorm1_input = self.norm1(x)
            attention_layer, distribution = self.attention(query=layernorm1_input, key=layernorm1_input,
                                                           value=layernorm1_input)  # seq_len, batch_size, hidden_dim
            first_sub_layer = self.dropout1(attention_layer) + x  # seq_len, batch_size, hidden_dim

            layernorm2_input = self.norm2(first_sub_layer)
            output = self.dropout2(self.output(layernorm2_input)) + first_sub_layer

        elif self.norm_type == 'post':
            ### PostLN code
            attention_layer, distribution = self.attention(query=x, key=x, value=x)  # seq_len, batch_size, hidden_dim
            first_sub_layer = self.norm1(self.dropout1(attention_layer) + x)  # seq_len, batch_size, hidden_dim
            output = self.norm2(self.dropout2(self.output(first_sub_layer)) + first_sub_layer)

        else:
            raise NotImplementedError

        # ### No Layernorm version
        # first_sub_layer =  self.dropout1(attention_layer) + x # seq_len, batch_size, hidden_dim
        # output =  self.dropout2(self.output(first_sub_layer)) + first_sub_layer

        # Equivalent to the below one.
        # output = self.norm2( self.dropout2( self.output(first_sub_layer.permute(1,0,2)).permute(1,0,2) ) + first_sub_layer )

        assert output.shape == input_shape
        return output, distribution


class BaseModel(nn.Module):
    def __init__(
            self,
            hidden_dim: int, num_layers: int,
            norm_type='post',
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__()

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=dropout, norm_type=norm_type, **kwargs) \
                    for _ in range(num_layers)]

        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        out = x
        distribution = []

        for encoder in self.encoders:
            out, distribution = encoder(out)

        return out, distribution