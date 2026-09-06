"""The rotate attention layer performs all the query key value projections and
output projections leaving the implementation of the attention to the inner
attention module.
"""

from torch.nn import Linear, Module

from fast_transformers.attention import AttentionLayer
from fast_transformers.events import EventDispatcher, QKVEvent
from .rotary import RotaryEmbedding, apply_rotary_pos_emb

class RotateAttentionLayer(AttentionLayer):
    """Rotate attention layer inherits from fast_transformer attention layer. 
        The only thing added is an Embedding encoding, for more information
        on the attention layer see the fast_transformers code
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(RotateAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using Rotation Embedding')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected. 
        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape #64 133 768
        _, S, _ = keys.shape #64 133 768
        H = self.n_heads #12

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1) #torch.Size([64, 133, 12, 64])
        keys = self.key_projection(keys).view(N, S, H, -1) #torch.Size([64, 133, 12, 64])
        cos, sin = self.rotaryemb(queries) #torch.Size([1, 133, 1, 64]) torch.Size([1, 133, 1, 64])
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin) #torch.Size([64, 133, 12, 64])
        values = self.value_projection(values).view(N, S, H, -1) #torch.Size([64, 133, 12, 64])
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1) #torch.Size([64, 133, 768])

        # Project the output and return
        return self.out_projection(new_values)
