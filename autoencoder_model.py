import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lion_pytorch import Lion
from time import time
from abc import ABC, abstractmethod

from mamba_model import ModelArgs, RMSNorm, ResidualBlock


class AbstractInterpolateModel(nn.Module, ABC):
    def __init__(self, input_dim, latent_dim, output_dim, num_layers, is_attention, hidden_multiplier, sequence_length, sparsity, return_all):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class InterpolateModel(AbstractInterpolateModel):
    def __init__(self, input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2, sequence_length=3,
                 sparsity=10, return_all=False):
        # super().__init__(input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2,
        #          sequence_length=3,
        #          sparsity=10, return_all=False)
        super().__init__(input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2, sequence_length=3,
                 sparsity=10, return_all=False)
        self.sparsity = sparsity
        self.sequence_length = sequence_length
        self.encoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_multiplier * 8, 8, batch_first=True, activation=F.leaky_relu),
            num_layers
        )

        self.linear_in = nn.Linear(input_dim, hidden_multiplier * 8)
        self.linear_out = nn.Linear(hidden_multiplier * 8, output_dim)
        self.return_all = return_all

    def forward(self, x):
        z = F.leaky_relu(self.linear_in(x))
        # create masking for autoregressive model
        # implement this for the future
        mask = torch.triu(torch.ones(z.shape[1]+1, z.shape[1]+1))
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask[1:, :-1]
        # mask=None

        z = self.encoder(z, z, tgt_mask=mask)
        z = self.linear_out(z)

        if not self.return_all:
            reconstructed_x = z[:, -1, :]
        else:
            reconstructed_x = z
        return reconstructed_x


class MambaModel(AbstractInterpolateModel):
    def __init__(self, input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2,
                 sequence_length=3,
                 sparsity=10, return_all=False):
        super().__init__(input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2,
                         sequence_length=3,
                         sparsity=10, return_all=False)

        self.args = ModelArgs(input_dim=input_dim, d_model=hidden_multiplier*8, n_layer=num_layers, vocab_size=input_dim)
        self.embedding = nn.Linear(self.args.input_dim, self.args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(self.args) for _ in range(self.args.n_layer)])
        self.norm_f = RMSNorm(self.args.d_model)

        self.lm_head = nn.Linear(self.args.d_model, self.args.input_dim, bias=False)
        self.return_all = return_all

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        outputs = self.lm_head(x)

        if not self.return_all:
            outputs = outputs[:, -1, :]

        return outputs


class ProbeModel(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, output_dim, num_layers, is_attention=False, hidden_multiplier=2, sequence_length=3, sparsity=10, learning_rate=0.001, return_all=False, model_type='transformer'):
        """

        :param input_dim:
        :param latent_dim:
        :param output_dim:
        :param num_layers:
        :param is_attention:
        :param hidden_multiplier:
        :param sequence_length:
        :param sparsity:
        :param learning_rate:
        :param return_all:
        :param model_type:  ["transformer", "mamba"]
        """
        super().__init__()
        self.sequence_length = sequence_length

        self.model_dict = {
            'transformer': InterpolateModel,
            'mamba': MambaModel
        }

        self.model = self.model_dict[model_type](input_dim, latent_dim, output_dim, num_layers,
                                 is_attention=is_attention, hidden_multiplier=hidden_multiplier,
                                      sequence_length=sequence_length, sparsity=sparsity, return_all=return_all)
        self.sparsity = sparsity
        self.learning_rate = learning_rate

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.start_time = time()

    def training_epoch_end(self, outputs) -> None:
        self.end_time = time()

        print(f'time taken: {self.end_time-self.start_time}')
    def training_step(self, batch, batch_idx):
        x, target, y = batch
        x = x.float()
        target = target.float()
        y = y.float()
        # exclude time variable
        x = x[:, :, 1:]
        target = target[..., 1:]
        prediction = self(x)
        loss = torch.sqrt(self.criterion(prediction, target))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.use_teacher_forcing = 1-(self.current_epoch / 100)

    def validation_step(self, batch, batch_idx):
        x, target, y = batch
        x = x.float()
        target = target.float()
        # exclude time variable
        x = x[:, :, 1:]
        target = target[..., 1:]
        prediction = self(x)
        loss = torch.sqrt(self.criterion(prediction, target))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
        return [optimizer], [scheduler]
