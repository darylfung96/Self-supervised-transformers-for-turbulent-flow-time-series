import math
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import tempfile

from trainer_type import TrainerType
from autoencoder_model import ProbeModel
from data import TimeSeriesIntermediateDataset, TimeSeriesIntermediateSelfSupervisedDataset

class InterpolateType(TrainerType):
    def __init__(self, running_type, model_type):
        super().__init__()
        self.sparsity = None
        self.running_type = running_type
        self.model_type = model_type

    def generate_model(self, input_dim, latent_dim, output_dim, num_layers, is_attention, hidden_multiplier, sequence_length, sparsity, learning_rate, return_all=False, model_type='transformer'):
        self.sparsity = sparsity
        return ProbeModel(input_dim, latent_dim, output_dim, num_layers, is_attention, hidden_multiplier, sequence_length, sparsity=sparsity, learning_rate=learning_rate, return_all=return_all, model_type=model_type)

    def create_train_test_split(self, *arrays, test_size=0.2, shuffle=False):
        # create the train test lists holder
        all_train_test_arrays = [[] for _ in range(len(arrays) * 2)]

        # all the case length inside the array has to have the same length
        for current_case_index in range(len(arrays[0])):
            combined_arrays = []
            for array_index in range(len(arrays)):
                combined_arrays.append(arrays[array_index][current_case_index])

            value_combined_arrays = [combined_array['value'] for combined_array in combined_arrays]
            train_test_arrays = train_test_split(*value_combined_arrays, test_size=0.2, shuffle=False)

            for i in range(len(train_test_arrays)):
                all_train_test_arrays[i].append(train_test_arrays[i])

        return all_train_test_arrays

    def generate_dataloader(self, x_train, x_test, y_train, y_test, step_size, sequence_length, target_length):

        all_data = []
        all_y = []
        for i in range(len(x_train)):
            all_data.append(np.concatenate([x_train[i], x_test[i]], 0))
            all_y.append(np.concatenate([y_train[i], y_test[i]], 0))

        # set all the dataset as training dataset so the mean and normalize will be -1 and 1 as requested
        # all_dataset = TimeSeriesIntermediateDataset(all_data, all_target, all_y, sparsity=target_length)
        train_dataset = TimeSeriesIntermediateDataset(x_train, y_train, step_size=step_size, sequence_length=sequence_length, sparsity=target_length,
                                                     )
        test_dataset = TimeSeriesIntermediateDataset(x_test, y_test, step_size=step_size, sequence_length=sequence_length, sparsity=target_length,
                                                     )
        batch_size = min(64, x_test[0].shape[0])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # self.all_dataset = all_dataset
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        return train_dataloader, test_dataloader

    def _interpolate(self, dataset, model, input_shape=(11, 5)):

        model.eval()
        sequence_length = model.sequence_length

        all_times = []
        all_predictions = []
        all_interpolations = []
        all_ground_truths = []
        all_losses = []

        interpolated_data = dataset.data
        sequenced_interpolated_data = dataset.sequenced_data
        sequenced_actual_data = dataset.sequenced_actual_data
        target = dataset.target
        interpolated_target = dataset.interpolated_target
        actual_data = dataset.actual_data

        for next_case in dataset.next_case_index[1:]:
            current_interpolated_data = interpolated_data[:next_case]
            current_interpolated_data = current_interpolated_data[:, :, 1:]

            prediction = model(current_interpolated_data)

            predicted_interpolated_data = sequenced_interpolated_data.clone()[:next_case*dataset.step_size+sequence_length, 1:]

            for current_step in range(prediction.shape[0]):
                predicted_interpolated_data[(current_step*dataset.step_size)+sequence_length] = prediction[current_step]

            all_predictions.append(predicted_interpolated_data.detach().cpu().numpy())
            all_ground_truths.append(sequenced_actual_data[:next_case*dataset.step_size+sequence_length, 1:].detach().cpu().numpy())
            all_interpolations.append(sequenced_interpolated_data[:next_case*dataset.step_size+sequence_length, 1:])
            all_times.append(sequenced_interpolated_data[:next_case*dataset.step_size+sequence_length, 0:1].detach().cpu().numpy())

            all_losses.append(((predicted_interpolated_data-sequenced_actual_data[:next_case*dataset.step_size+sequence_length, 1:])**2).mean().detach().cpu().numpy())

        all_times = np.concatenate(all_times, 0)
        all_predictions = np.concatenate(all_predictions, 0)
        all_interpolations = np.concatenate(all_interpolations, 0)
        all_ground_truths = np.concatenate(all_ground_truths, 0)

        return all_times, all_predictions, all_interpolations, all_ground_truths, all_losses


    def plot_prediction(self, model, absmax, dataset_type='test_dataset', input_shape=(11, 5)):
        dataset_dict = {
            'test_dataset': self.test_dataset,
            'train_dataset': self.train_dataset
        }

        current_dataset = dataset_dict[dataset_type]

        model.eval()
        all_time, all_predictions, all_interpolations, all_ground_truths, loss = self._interpolate(current_dataset, model, input_shape=input_shape)
        actual_pred_index = [idx for idx in range(0, all_predictions.shape[0], current_dataset.sparsity)]

        all_predictions = np.reshape(all_predictions, [all_predictions.shape[0], *input_shape])
        all_interpolations = np.reshape(all_interpolations, [all_interpolations.shape[0], *input_shape])
        all_ground_truths = np.reshape(all_ground_truths, [all_ground_truths.shape[0], *input_shape])
        np.mean(np.sqrt((all_predictions-all_ground_truths)**2))
        absmax = absmax.reshape(input_shape)
        all_val_loss = []
        for x_pos in range(all_predictions.shape[1]):
            for z_pos in range(all_predictions.shape[2]):

                all_case_index = current_dataset.next_case_index
                end_index = 0
                for case, next_case_index in enumerate(range(len(all_case_index) - 1)):

                    start_index = end_index
                    end_index = all_case_index[next_case_index+1] * current_dataset.step_size + model.sequence_length

                    current_case_pred = all_predictions[start_index:end_index, x_pos, z_pos] * absmax[x_pos, z_pos]
                    current_case_interpolation = all_interpolations[start_index:end_index, x_pos, z_pos] * absmax[x_pos, z_pos]
                    current_case_target = all_ground_truths[start_index:end_index, x_pos, z_pos] * absmax[x_pos, z_pos]
                    current_case_time = all_time[start_index:end_index, 0]

                    loss = np.mean(np.sqrt((current_case_target - current_case_pred)**2))
                    all_val_loss.append([case, x_pos, z_pos, loss])

                    plt.title(f'Prediction vs ground truth of {x_pos + 1}, {z_pos + 1} position in case {case+1} (sparsity: {current_dataset.sparsity})')
                    plt.xlabel("time")
                    plt.ylabel("normalized pressure signal")
                    plt.plot(current_case_time, current_case_pred, color='green', linestyle="--", label='prediction')
                    plt.plot(current_case_time, current_case_interpolation, color='orange', linestyle="--", label='interpolation')
                    plt.plot(current_case_time, current_case_target, color='black', label='ground truths', alpha=0.8)
                    # plt.ylim(-1, 1)
                    # plt.scatter(current_case, actual_case_pred, color='green', label=f'ground truths (sparsity: {sparsity})')
                    plt.legend()

                    # log the prediction vs ground truth plot
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        plot_file_path = temp_file.name
                        plt.savefig(plot_file_path)
                        mlflow.log_artifact(plot_file_path)
                    os.remove(plot_file_path)
                    plt.clf()
                    plt.close()

                    os.makedirs(f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}', exist_ok=True)
                    df = pd.DataFrame(np.array([current_case_time, current_case_interpolation, current_case_pred, current_case_target]).T, columns=['time', 'spline', 'prediction', 'target'])
                    df.to_csv(f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv')
                    mlflow.log_artifact(
                        f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv',
                        artifact_path=f'Data/csv/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv')

        all_val_loss = np.array(all_val_loss)
        mlflow.log_table(pd.DataFrame({'case': all_val_loss[:, 0], 'x_pos': all_val_loss[:, 1], 'z_pos': all_val_loss[:, 2],
                          'loss': all_val_loss[:, 3]}), artifact_file=f'Data/table/{dataset_type}/{self.model_type}_{self.running_type}.csv')


class SelfSupervisedInterpolatedType(InterpolateType):
    def generate_dataloader(self, x_train, x_test, y_train, y_test, step_size, sequence_length, target_length, mask_prob=0.3):

        all_data = []
        all_y = []
        for i in range(len(x_train)):
            all_data.append(np.concatenate([x_train[i], x_test[i]], 0))
            all_y.append(np.concatenate([y_train[i], y_test[i]], 0))

        # set all the dataset as training dataset so the mean and normalize will be -1 and 1 as requested
        # all_dataset = TimeSeriesIntermediateDataset(all_data, all_target, all_y, sparsity=target_length)
        train_dataset = TimeSeriesIntermediateSelfSupervisedDataset(x_train, y_train, step_size=step_size, sequence_length=sequence_length, sparsity=target_length,
                                                     mask_prob=mask_prob)
        test_dataset = TimeSeriesIntermediateSelfSupervisedDataset(x_test, y_test, step_size=step_size, sequence_length=sequence_length, sparsity=target_length,
                                                     mask_prob=mask_prob)
        batch_size = min(64, x_test[0].shape[0])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # self.all_dataset = all_dataset
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        return train_dataloader, test_dataloader

    def _interpolate(self, dataset, model, input_shape=(11, 5)):

        model.eval()
        sequence_length = model.sequence_length

        all_times = []
        all_predictions = []
        all_ground_truths = []
        all_losses = []

        interpolated_data = dataset.data
        target = dataset.target

        for next_case in dataset.next_case_index[1:]:
            current_interpolated_data = interpolated_data[:next_case]
            current_interpolated_data = current_interpolated_data[:, :, 1:]

            prediction = model(current_interpolated_data)

            all_predictions.append(prediction.detach().cpu().numpy())
            all_ground_truths.append(target[:next_case, :, 1:].detach().cpu().numpy())
            all_times.append(target[:next_case, :, 0:1].detach().cpu().numpy())

            all_losses.append(((prediction-target[:next_case, :, 1:])**2).mean().detach().cpu().numpy())

        all_times = np.concatenate(all_times, 0)
        all_predictions = np.concatenate(all_predictions, 0)
        all_ground_truths = np.concatenate(all_ground_truths, 0)

        return all_times, all_predictions, all_ground_truths, all_losses

    def plot_prediction(self, model, absmax, dataset_type='test_dataset', input_shape=(11, 5)):
        dataset_dict = {
            'test_dataset': self.test_dataset,
            'train_dataset': self.train_dataset
        }

        current_dataset = dataset_dict[dataset_type]

        model.eval()
        all_time, all_predictions, all_ground_truths, loss = self._interpolate(current_dataset, model, input_shape=input_shape)
        actual_pred_index = [idx for idx in range(0, all_predictions.shape[0], current_dataset.sparsity)]

        all_predictions = np.reshape(all_predictions, [all_predictions.shape[0], all_predictions.shape[1], *input_shape])
        all_ground_truths = np.reshape(all_ground_truths, [all_ground_truths.shape[0], all_ground_truths.shape[1], *input_shape])
        np.mean(np.sqrt((all_predictions-all_ground_truths)**2))
        absmax = absmax.reshape(input_shape)
        all_val_loss = []
        for x_pos in range(all_predictions.shape[2]):
            for z_pos in range(all_predictions.shape[3]):

                all_case_index = current_dataset.next_case_index
                end_index = 0
                for case, next_case_index in enumerate(range(len(all_case_index) - 1)):

                    start_index = end_index
                    end_index = all_case_index[next_case_index+1]

                    current_case_pred = all_predictions[start_index:end_index, :, x_pos, z_pos] * absmax[x_pos, z_pos]
                    current_case_target = all_ground_truths[start_index:end_index, :, x_pos, z_pos] * absmax[x_pos, z_pos]
                    current_case_time = all_time[start_index:end_index, :, 0]

                    current_case_pred = current_case_pred.reshape(-1)
                    current_case_target = current_case_target.reshape(-1)
                    current_case_time = current_case_time.reshape(-1)

                    loss = np.mean(np.sqrt((current_case_target - current_case_pred)**2))
                    all_val_loss.append([case, x_pos, z_pos, loss])

                    plt.title(f'Prediction vs ground truth of {x_pos + 1}, {z_pos + 1} position in case {case+1} (sparsity: {current_dataset.sparsity})')
                    plt.xlabel("time")
                    plt.ylabel("normalized pressure signal")
                    plt.plot(current_case_time, current_case_pred, color='green', linestyle="--", label='prediction')
                    plt.plot(current_case_time, current_case_target, color='black', label='ground truths', alpha=0.8)
                    # plt.ylim(-1, 1)
                    # plt.scatter(current_case, actual_case_pred, color='green', label=f'ground truths (sparsity: {sparsity})')
                    plt.legend()

                    # log the prediction vs ground truth plot
                    plot_file_path = os.path.join(f'Data/{case}_{x_pos}_{z_pos}.png')
                    plt.savefig(plot_file_path)
                    mlflow.log_artifact(plot_file_path, artifact_path=f'Data/plots/{dataset_type}/{case}_{x_pos}_{z_pos}.png')
                    os.remove(plot_file_path)
                    plt.clf()
                    plt.close()

                    os.makedirs(f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}', exist_ok=True)
                    df = pd.DataFrame(np.array([current_case_time, current_case_pred, current_case_target]).T, columns=['time', 'prediction', 'target'])
                    df.to_csv(f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv')
                    mlflow.log_artifact(f'Data/temp/spline_{self.model_type}_{self.running_type}/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv',
                                        artifact_path=f'Data/csv/{dataset_type}/{case}/{x_pos}/{z_pos}/{current_dataset.sparsity}_{current_dataset.step_size}_{model.sequence_length}.csv')

        all_val_loss = np.array(all_val_loss)
        mlflow.log_table(pd.DataFrame({'case': all_val_loss[:, 0], 'x_pos': all_val_loss[:, 1], 'z_pos': all_val_loss[:, 2], 'loss': all_val_loss[:, 3]}),
                         artifact_file=f'Data/table/{dataset_type}/{self.model_type}_{self.running_type}.csv')
