import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=11, out_channels=5):
        super(Unet, self).__init__()

        self.dconv_down1 = DoubleConv(in_channels, 1)
        self.dconv_down2 = DoubleConv(1, 2)
        self.dconv_down3 = DoubleConv(2, 4)
        self.dconv_down4 = DoubleConv(4, 8)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dconv_up3 = DoubleConv(4, 4)
        self.dconv_up2 = DoubleConv(1 + 4, 2)

        self.conv_last = nn.Conv2d(2, out_channels, kernel_size=1)

    def forward(self, x):

        # Downward path
        x1 = self.dconv_down1(x)
        x2 = self.maxpool(x1)
        x3 = self.dconv_down2(x2)
        x4 = self.maxpool(x3)

        # Upward path
        x = nn.functional.interpolate(x3, scale_factor=3, mode='bilinear', align_corners=True)
        # since x3 has odd dimensions, we create x having extra dimensions to account for that
        # since scale factor always produce even dimension
        x = torch.cat([x[:, :, :x3.shape[-2], :x3.shape[-1]], x3], dim=1)
        x = self.dconv_up3(x)

        x = nn.functional.interpolate(x, scale_factor=3, mode='bilinear', align_corners=True)
        x = torch.cat([x[:, :, :x1.shape[-2], :x1.shape[-1]], x1], dim=1)
        x = self.dconv_up2(x)

        x = self.conv_last(x)

        return x


class ProbeModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = Unet(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

    def forecast(self, x):
        lstm_out, hidden = self.lstm(x)
        out = self.fc(lstm_out[:, :, :])
        return out, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.use_teacher_forcing = 1-(self.current_epoch / 100)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer
