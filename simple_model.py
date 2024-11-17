import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LinearModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_multiplier, output_dim):
        super().__init__()
        self.first_blocks = nn.ModuleList([
            nn.Linear(input_dim, 32 * hidden_multiplier),
            nn.LayerNorm(32 * hidden_multiplier),
            nn.ReLU(),
            nn.Linear(32 * hidden_multiplier, 16 * hidden_multiplier),
            nn.LayerNorm(16 * hidden_multiplier),
            nn.ReLU()
        ])

        self.encoder_y = nn.Sequential(nn.Linear(55, 8 * hidden_multiplier),
                                       nn.ReLU(),
                                       nn.Linear(8 * hidden_multiplier, 8 * hidden_multiplier),
                                       nn.ReLU(),
                                       nn.Linear(8 * hidden_multiplier, 16 * hidden_multiplier),
                                       nn.ReLU())

        self.second_blocks = nn.ModuleList([
            nn.Linear(16 * hidden_multiplier + 16 * hidden_multiplier, 16 * hidden_multiplier),
            nn.LayerNorm(16 * hidden_multiplier),
            nn.ReLU(),
            nn.Linear(16 * hidden_multiplier, output_dim)
        ])

    def forward(self, x, y):
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        for block in self.first_blocks:
            x = block(x)

        encoded_y = self.encoder_y(y)

        encoded_z = torch.cat([x, encoded_y], 1)

        for block in self.second_blocks:
            encoded_z = block(encoded_z)

        encoded_z = encoded_z.reshape(encoded_z.shape[0], 11, 5)
        return encoded_z

    def training_step(self, batch, batch_idx):
        x, target, y = batch
        y_hat = self(x, y)
        loss = F.mse_loss(y_hat, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, y = batch
        prediction = self(x, y)
        loss = F.mse_loss(prediction, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
