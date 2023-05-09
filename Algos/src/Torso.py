from typing import *
import torch
from torch import nn

#Transformer Bloack which converts Tensors to grids (SxSxS) -> (SxS)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tensor passes through layer normalisation operation
        x = self.norm1(x)
        # Tensor passes through multi-head self-attention layer
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AttentiveModels(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.block1 = TransformerBlock(embed_dim)
        self.block2 = TransformerBlock(embed_dim)
        self.block3 = TransformerBlock(embed_dim)

    def forward(self, grids):
        grids = [g.clone() for g in grids]
        for i, j, block in [(0, 1, self.block1), (2, 0, self.block2), (1, 2, self.block3)]:
            x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
            x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)

            S = grids[i].shape[2]
            assert grids[j].shape[1] == S
            assert x.shape[2] == 2 * S

            grids[i] = x[:, :, :S]
            grids[j] = x[:, :, S:].transpose(1, 2)

        return grids


class Torso(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int], embed_dim: int = 256, num_attn_models: int = 4):
        super().__init__()
 
        self.fc1 = nn.Linear(input_size[-1], embed_dim)
        self.fc2 = nn.Linear(input_size[-2], embed_dim)
        self.fc3 = nn.Linear(input_size[-3], embed_dim)

        self.attn_models = nn.ModuleList([AttentiveModels() for _ in range(num_attn_models)])

        self.input_size = input_size
        self.embed_dim = embed_dim

    def forward(self, s: torch.Tensor):
        assert s.ndim == 4
        assert s.shape[-3:] == self.input_size
        # generating grids over s, permuting dimentions of s
        grids = self.fc1(s), self.fc2(s.permute((0, 3, 1, 2))), self.fc3(s.permute((0, 2, 3, 1)))
        #extracting the features from grids using transformer blocks and Attentive models
        for attn_models in self.attn_models:
            grids = attn_models(grids)
        #generating embedding by stacking the updated grids.
        e = torch.stack(grids, dim=1).flatten(1, -2)  # (bs, 3 * S * S, c)
        return torch.max(e, dim=1).values  # (bs, c), global max pool instead of cross attn