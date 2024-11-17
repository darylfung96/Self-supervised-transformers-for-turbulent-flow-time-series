import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from trainer_type import TrainerType
from data import TimeSeriesForecastDataset
from time_model import LSTMModel


class ForecastType(TrainerType):
    def __init__(self):
        super().__init__()
        self.sequence_length = 30

    def generate_model(self, input_dim, latent_dim, output_dim, num_layers, is_attention, hidden_multiplier, target_length):
        self.target_length = target_length
        return LSTMModel(input_dim, hidden_multiplier, output_dim, num_layers, self.target_length, is_attention)

    def create_train_test_split(self, *arrays, test_size=0.2, shuffle=False):
        train_test_arrays = []

        for array in arrays:
            current_train_array = []
            current_test_array = []
            for current_case_data in array:
                current_test_size = int(current_case_data.shape[0] * test_size)
                train = current_case_data[:current_case_data.shape[0]-current_test_size]
                test = current_case_data[current_case_data.shape[0]-current_test_size:]

                current_train_array.append(train)
                current_test_array.append(test)

            train_test_arrays.append(current_train_array)
            train_test_arrays.append(current_test_array)

        return train_test_arrays

    def generate_dataloader(self, x_train, x_test, y_train, y_test):
        train_dataset = TimeSeriesForecastDataset(x_train, y_train, self.sequence_length, self.target_length)
        test_dataset = TimeSeriesForecastDataset(x_test, y_test, self.sequence_length, self.target_length,
                                                 scaler=train_dataset.scaler,
                                                     y_scaler=train_dataset.y_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        return train_dataloader, test_dataloader

    def plot_prediction(self, model, wandb_logger):

        # plot the predictions and ground truths

        x, target, y = self.test_dataset[0]
        x = torch.tensor(x).float().unsqueeze(0)
        y = torch.tensor(y).float().unsqueeze(0)
        target = torch.tensor(target).float().unsqueeze(0)
        pred = model(x, y)

        all_predictions = pred
        all_targets = target

        for idx in range(1, len(self.test_dataset)):
            x, target, y = self.test_dataset[idx]
            x = torch.tensor(x).float().unsqueeze(0)
            y = torch.tensor(y).float().unsqueeze(0)
            target = torch.tensor(target).float().unsqueeze(0)

            pred = model(x, y)
            all_predictions = torch.cat([all_predictions, pred], 1)
            all_targets = torch.cat([all_targets, target], 1)

        all_predictions = all_predictions.reshape(all_predictions.shape[0], all_predictions.shape[1], 11, 5)
        all_targets = all_targets.reshape(all_targets.shape[0], all_targets.shape[1], 11, 5)

        all_predictions = self.test_dataset.scaler.inverse_transform(np.reshape(all_predictions[0].detach().numpy(), [-1, 55]))
        all_targets = self.test_dataset.scaler.inverse_transform(np.reshape(all_targets[0].detach().numpy(), [-1, 55]))
        all_predictions = np.reshape(all_predictions, [all_predictions.shape[0], 11, 5])
        all_targets = np.reshape(all_targets, [all_targets.shape[0], 11, 5])
        for x_pos in range(all_predictions.shape[1]):
            for z_pos in range(all_predictions.shape[2]):

                all_case_index = [0] + self.test_dataset.next_case_index
                for case, next_case_index in enumerate(range(len(all_case_index)-1)):

                    current_pred = all_predictions[all_case_index[next_case_index]:all_case_index[next_case_index+1],
                                   x_pos, z_pos]
                    current_target = all_targets[all_case_index[next_case_index]:all_case_index[next_case_index+1],
                                     x_pos, z_pos]

                    plt.title(f'last 100 Prediction vs ground truth of {x_pos + 1}, {z_pos + 1} position in case {case+1}')
                    plt.xlabel("time")
                    plt.ylabel("normalized pressure signal")
                    plt.plot(np.arange(current_pred.shape[0]), current_pred, color='green', label='prediction')
                    plt.plot(np.arange(current_target.shape[0]), current_target, color='black', label='ground truths')
                    plt.ylim(ymin=0)
                    plt.legend()
                    wandb_logger.log_image(key=f"case: {case+1} x_pos: {x_pos+1} z_pos: {z_pos+1}", images=[plt])
                    plt.clf()
                    plt.close()

