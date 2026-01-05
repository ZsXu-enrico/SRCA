"""Learnable aggregation modules for token-level features."""

import torch
import torch.nn as nn


class LearnableWeightedPooling(nn.Module):
    """Learn a weight for each token position, then weighted sum.

    Converts [batch, seq_len, hidden] -> [batch, hidden]
    """

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        # Learnable weights for each position
        self.weights = nn.Parameter(torch.ones(seq_len) / seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden]
        Returns:
            [batch, hidden]
        """
        # Normalize weights to sum to 1
        weights = torch.softmax(self.weights, dim=0)  # [seq_len]

        # Weighted sum over sequence dimension
        # weights: [seq_len] -> [1, seq_len, 1]
        weights = weights.view(1, self.seq_len, 1)
        output = (x * weights).sum(dim=1)  # [batch, hidden]

        return output


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence.

    Learn to attend to important tokens.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Query vector for attention
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden]
        Returns:
            [batch, hidden]
        """
        # Compute attention scores
        # x: [batch, seq_len, hidden]
        # query: [hidden]
        scores = torch.matmul(x, self.query)  # [batch, seq_len]

        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]

        # Weighted sum
        # attn_weights: [batch, seq_len, 1]
        output = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]

        return output


class SeparateAggregation(nn.Module):
    """Separate aggregation modules for mashup and API.

    This allows mashup and API to learn different aggregation strategies.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        aggregation_type: str = 'attention'
    ):
        """
        Args:
            seq_len: Sequence length (e.g., 50 tokens)
            hidden_dim: Hidden dimension (e.g., 1536)
            aggregation_type: 'attention' or 'weighted'
        """
        super().__init__()

        if aggregation_type == 'attention':
            self.mashup_aggregator = AttentionPooling(hidden_dim)
            self.api_aggregator = AttentionPooling(hidden_dim)
        elif aggregation_type == 'weighted':
            self.mashup_aggregator = LearnableWeightedPooling(seq_len)
            self.api_aggregator = LearnableWeightedPooling(seq_len)
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")

        self.aggregation_type = aggregation_type

    def forward_mashup(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate mashup token features."""
        return self.mashup_aggregator(x)

    def forward_api(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate API token features."""
        return self.api_aggregator(x)

    def forward(self, mashup_tokens: torch.Tensor, api_tokens: torch.Tensor):
        """
        Args:
            mashup_tokens: [num_mashups, seq_len, hidden]
            api_tokens: [num_apis, seq_len, hidden]

        Returns:
            mashup_features: [num_mashups, hidden]
            api_features: [num_apis, hidden]
        """
        mashup_features = self.forward_mashup(mashup_tokens)
        api_features = self.forward_api(api_tokens)
        return mashup_features, api_features


if __name__ == "__main__":
    # Test
    print("Testing Aggregation Modules...")

    batch_size = 4
    seq_len = 50
    hidden = 1536

    # Create fake token features
    tokens = torch.randn(batch_size, seq_len, hidden)

    # Test weighted pooling
    print("\n1. Learnable Weighted Pooling:")
    weighted_pool = LearnableWeightedPooling(seq_len)
    output = weighted_pool(tokens)
    print(f"   Input: {tokens.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Weights (first 10): {weighted_pool.weights[:10].data}")

    # Test attention pooling
    print("\n2. Attention Pooling:")
    attn_pool = AttentionPooling(hidden)
    output = attn_pool(tokens)
    print(f"   Input: {tokens.shape}")
    print(f"   Output: {output.shape}")

    # Test separate aggregation
    print("\n3. Separate Aggregation:")
    mashup_tokens = torch.randn(10, seq_len, hidden)
    api_tokens = torch.randn(5, seq_len, hidden)

    sep_agg = SeparateAggregation(seq_len, hidden, aggregation_type='attention')
    mashup_feat, api_feat = sep_agg(mashup_tokens, api_tokens)
    print(f"   Mashup: {mashup_tokens.shape} -> {mashup_feat.shape}")
    print(f"   API: {api_tokens.shape} -> {api_feat.shape}")

    print("\n✓ All tests passed!")
