from typing import *
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from Heads import Value_Head,Policy_Head
from Torso import Torso
from Environment import State


class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[Tuple[State, np.ndarray, float]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, v = self.examples[idx]
        return np.float32(state.tensor), np.float32(pi), np.float32(v)


class DataModule(pl.LightningDataModule):
    def __init__(self, examples: Sequence[Tuple[State, np.ndarray, float]], batch_size: int):
        super().__init__()

        self.dataset = Dataset(examples)
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)

class NeuralNet(pl.LightningModule):
    def __init__(self, input_size: Tuple[int, int, int], num_actions: int,
                 embed_dim: int = 256, u_q: float = 0.75, lr: float = 1e-2):
        super().__init__()

        self.torso = Torso(input_size, embed_dim)
        self.policy_head = Policy_Head(num_actions, embed_dim)
        self.value_head = Value_Head(embed_dim)

        self.save_hyperparameters()

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.torso(s)
        return self.policy_head(e), self.value_head(e)

    def Quantile_loss(self, quantiles, target): # Quantile loss used in the value head. 
        # quantiles are the predicted values of the quantiles at equally spaced intervals, target is the ground truth.
        num_quantiles = quantiles.shape[1]
        tau = (torch.arange(num_quantiles).to(quantiles) + 0.5) / num_quantiles

        target = target.unsqueeze(1)
        weights = torch.where(quantiles > target, tau, 1 - tau)
        return torch.mean(weights * F.huber_loss(quantiles, target, reduction='none'))

    def training_step(self, batch, batch_idx=None):
        s, pi, v = batch

        logits, quantiles = self.forward(s)

        policy_loss = F.cross_entropy(logits, pi)
        self.log('policy_loss', policy_loss, on_epoch=True)

        value_loss = self.Quantile_loss(quantiles, v)
        self.log('value_loss', value_loss, on_epoch=True)

        return policy_loss + value_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        self.eval()

        s = torch.tensor(state.tensor, dtype=torch.float32, device=self.device)
        s = s.unsqueeze(0)  # add batch dim
        logits, quantiles = self.forward(s)
        logits, quantiles = logits.squeeze(), quantiles.squeeze()  # remove batch dims
        p = torch.softmax(logits, dim=0)
        v = torch.mean(quantiles[int(len(quantiles) * self.hparams.u_q):])
        return p.data.cpu().numpy(), v.item()

    def Value_Risk_Management(Q,Uq: int = 0.75): #Obtaining the predicted value at inference
        return torch.mean(Q[int(Q.shape[1] * Uq):])