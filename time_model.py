import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_multiplier, output_dim, num_layers, target_length, is_attention):
        super().__init__()

        self.target_length = target_length
        self.is_attention = is_attention
        self.hidden_multiplier = hidden_multiplier

        self.pre_block = nn.Sequential(
            nn.Linear(input_dim, 4 * hidden_multiplier),
            nn.LayerNorm(4 * hidden_multiplier, 4 * hidden_multiplier),
            nn.ReLU(),
        )

        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, 4 * hidden_multiplier),
            nn.ReLU(),
            nn.Linear(4 * hidden_multiplier, 8 * hidden_multiplier),
            nn.ReLU(),
            nn.Linear(8 * hidden_multiplier, 4 * hidden_multiplier)
        )

        self.lstm = nn.LSTM(4 * hidden_multiplier * 2, 4 * hidden_multiplier, num_layers)

        if self.is_attention:
            self.pre_attention_layer = nn.Sequential(
                nn.LayerNorm(4 * hidden_multiplier),
                nn.ReLU()
            )
        self.attention_layer = nn.MultiheadAttention(4 * hidden_multiplier, 8)

        self.post_block = nn.Linear(4 * hidden_multiplier, output_dim)

    def forward(self, x, y):
        output = self.pre_block(x)
        encoded_y = self.y_encoder(y)
        output = torch.cat([output, encoded_y], -1)
        lstm_output, (h, c) = self.lstm(output)

        if self.is_attention:
            lstm_output = self.pre_attention_layer(lstm_output)
            lstm_output, _ = self.attention_layer(lstm_output, lstm_output, lstm_output)
            lstm_output = F.layer_norm(lstm_output, [4 * self.hidden_multiplier])

        all_outputs = lstm_output[:, -1:, :]
        for i in range(self.target_length-1):

            output = torch.cat([lstm_output[:, -1:, :], encoded_y[:, -1:, :]], -1)
            lstm_output, (h, c) = self.lstm(output, (h[:, -1:, :], c[:, -1:, :]))

            if self.is_attention:
                lstm_output = self.pre_attention_layer(lstm_output)
                lstm_output, _ = self.attention_layer(lstm_output, lstm_output, lstm_output)
                lstm_output = F.layer_norm(lstm_output, [4 * self.hidden_multiplier])

            all_outputs = torch.cat([all_outputs, lstm_output[:, -1:, :]], 1)

        prediction = self.post_block(all_outputs)
        return prediction

    def training_step(self, batch, batch_idx):
        x, target, y = batch
        x = x.float()
        target = target.float()
        y = y.float()

        y_hat = self(x, y)
        loss = F.mse_loss(y_hat, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, y = batch
        x = x.float()
        target = target.float()
        y = y.float()

        prediction = self(x, y)
        loss = F.mse_loss(prediction, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
