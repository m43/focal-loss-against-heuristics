# heavily inspired from transformers.models.distilbert.modeling_distilbert.Embeddings
import torch
from torch import nn

from packaging import version

from transformers.models.distilbert import modeling_distilbert


class NliEmbeddings(nn.Module):
    def __init__(self, src: modeling_distilbert.Embeddings):
        super().__init__()

        self.inner = src
        # should we do any particular weight init?
        self.segment_embeddings = nn.Embedding(2, src.word_embeddings.embedding_dim)

        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "position_ids", torch.arange(self.inner.position_embeddings.num_embeddings).expand((1, -1)),
                persistent=False
            )

    def forward(self, input_ids, token_type_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
            token_type_ids: torch.tensor(bs, max_seq_length) The id of the token types

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self.inner, "position_ids"):
            position_ids = self.inner.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.inner.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.inner.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        segment_embeddings = self.segment_embeddings(token_type_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings + segment_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.inner.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.inner.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings
