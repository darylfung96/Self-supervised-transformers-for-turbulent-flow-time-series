import numpy as np
import os
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_probe(case, x, z, x_coarse=None):
    folder_dir = 'Data'
    y_filename = os.path.join(folder_dir, 'y_deformations.xlsx')
    df_y = pd.read_excel(y_filename)

    current_pos, current_y = _load_probe(folder_dir, df_y, case, x, z, x_coarse)
    x = current_pos['Pressure'].to_numpy()

    scaler = StandardScaler()
    x_mean = 0
    x_abs_max = 0
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = x - x_mean
    x = x / x_std
    # x_abs_max = np.max(abs(x))
    # x = x / x_abs_max
    x = np.expand_dims(x, 1)
    # x = scaler.fit_transform(np.expand_dims(x, 1))
    # add the time back
    x = np.concatenate([np.expand_dims(np.array(current_pos['Time']), 1), x], 1)

    # y = current_y.to_numpy()
    # y_mean = y.mean()
    # y = y - y_mean
    # y_abs_max = np.max(abs(y))
    # y = y / y_abs_max

    y_mean = current_y
    y_abs_max = current_y
    current_y = np.tile(current_y, x.shape)

    x_dict = {"value": x, "mean": x_mean, "abs_max": x_std}
    y_dict = {"value": current_y, "mean": y_mean, "abs_max": y_abs_max}

    return [x_dict], [y_dict]

def _load_probe(folder_dir, df_y, case, x, z, x_coarse=None):
    filename = os.path.join(folder_dir, f'case{"{:02}".format(case)}', f'probe({"{:02}".format(x)},{z}).csv')
    current_pos = pd.read_csv(filename, header=None, names=['Time', 'Pressure'])
    # convert all columns to float
    current_pos = current_pos.astype(float)
    # round to specified coarseness
    if x_coarse is not None:
        current_pos['Time'] = current_pos['Time'].round(x_coarse)
    current_pos = current_pos[~current_pos['Time'].duplicated(keep='first')]

    # get the x_pos and z_pos of the y_deformations
    current_y_position = df_y[(df_y.iloc[:, 0] == x) & (df_y.iloc[:, 1] == z)]
    # get y based on case
    current_y = current_y_position[f'Case {case}'].iloc[0]

    return current_pos, current_y

def load_csv(case=1, x_coarse=None, input_shape=(11, 5)):
    """
        Load csv file with has two columns, the first column is time and the second column is the pressure signal.
        The y_filename will contain the height of the positions.
    :param x_pos:           The position of x-axis.
    :param z_pos:           The position of y-axis.
    :param filename:        The filename of the csv file with pressure signal.
    :param y_filename:      The filename of the csv file with height of positions.
    :param case:            The case to get for the dataset
    :param x_coarse:        The quantity of roughness for x. For example, x varies in value by increments of 0.000001. x_coarse
                            adjust it such that x changes value according to the decimal point. If x_coarse is 5, then x_pos will be
                            an interval that changes with a decimal point of 5.
    :param sparsity:        The sparsity of the data
    :param type:            "intermediate", "forecast"
    :return:
    """
    assert 0 <= case <= 11
    folder_dir = 'Data'

    df = pd.DataFrame()
    y_filename = os.path.join(folder_dir, 'y_deformations.xlsx')
    df_y = pd.read_excel(y_filename)
    all_y = {}
    if case == 0:
        for x in range(1, 10):
            current_pos, current_y = _load_probe(folder_dir, df_y, case, x, 1, x_coarse)
            df[f'{x},1'] = current_pos['Pressure']
            all_y[f'{x},1,y'] = current_y
    else:
        for x in range(1, 12):
            for z in range(1, 6):
                current_pos, current_y = _load_probe(folder_dir, df_y, case, x, z, x_coarse)
                df[f'{x},{z}'] = current_pos['Pressure']
                all_y[f'{x},{z},y'] = current_y

    # # create a pressure target that shows the next pressure from Pressure column
    # df['Pressure_target'] = df['Pressure'].shift(-1)
    # # remove the last row, as the last row does not have target
    # df = df[:-1]
    df_all_y = pd.DataFrame(all_y, index=['y'])

    x = df.to_numpy()
    x_mean = np.mean(x, (0))
    x_std = np.std(x, (0))
    x = x - x_mean
    # x_abs_max = np.max(abs(x), 0)
    # x = x / x_abs_max
    x = x / x_std
    # add the time back
    x = np.concatenate([np.expand_dims(np.array(current_pos['Time']), 1), x], 1)

    y = df_all_y.to_numpy()
    y_mean = y.mean()
    y = y - y_mean
    y_abs_max = np.max(abs(y))
    y = y / y_abs_max

    y = np.reshape(y, [1, *input_shape])
    y = np.repeat(y, x.shape[0], axis=0)

    x_dict = {"value": x, "mean": x_mean, "abs_max": x_std}
    y_dict = {"value": y, "mean": y_mean, "abs_max": y_abs_max}

    return [x_dict], [y_dict]


def load_all_case():
    all_x = []
    all_y = []
    for i in range(1, 7):
        current_x, current_y = load_csv(case=i)
        all_x.append(current_x[0])
        all_y.append(current_y[0])

    return all_x, all_y


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, seq_length):
        self.data = data
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = (idx + 1) * self.seq_length
        seq = self.data[start_idx:end_idx, :]
        labels = self.labels[start_idx:end_idx]
        return seq, labels


class TimeSeriesIntermediateDataset(Dataset):
    def __init__(self, data, height, step_size=2, sequence_length=3, sparsity=2):
        """

        :param data:                the pressure signal of the location
        :param height:              the height of the probe coord
        :param sparsity:
        :param scaler:
        :param y_scaler:
        :param normalize:
        """
        self.sequence_length = sequence_length
        self.step_size = step_size
        # create interpolation model based on each case
        interpolated_datas = []
        sequenced_interpolated_datas = []
        interpolated_target_datas = []
        sequenced_actual_datas = []
        actual_datas = []
        targets = []
        actual_heights = []
        self.interpolation_models = []
        for idx, current_data in enumerate(data):
            original_index = set(range(0, current_data.shape[0], sparsity))
            interpolate_index = set(range(data[0].shape[0])) - original_index

            original_index = sorted(original_index)
            interpolate_index = sorted(interpolate_index)

            # remove out of bound indexes
            while interpolate_index[-1] > original_index[-1]:
                interpolate_index = interpolate_index[:-1]

            new_data = current_data[original_index]
            interpolation_model = interp1d(new_data[:, 0], new_data[:, 1:], axis=0, kind='cubic')
            self.interpolation_models.append(interpolation_model)

            sequenced_interpolated_data = interpolation_model(current_data[:original_index[-1], 0])
            sequenced_interpolated_data = np.concatenate([current_data[:original_index[-1], 0:1], sequenced_interpolated_data], 1)
            interpolated_target_data = np.concatenate([np.expand_dims(sequenced_interpolated_data[i + sequence_length], 0) for i in
                                        range(0, sequenced_interpolated_data.shape[0] - sequence_length - 1, step_size)])
            interpolated_target_datas.append(interpolated_target_data)
            interpolated_data = [np.expand_dims(sequenced_interpolated_data[i:i + sequence_length], 0) for i in
             range(0, sequenced_interpolated_data.shape[0] - sequence_length-1, step_size)]
            interpolated_data = np.concatenate(interpolated_data)
            sequenced_interpolated_datas.append(sequenced_interpolated_data)
            interpolated_datas.append(interpolated_data)

            sequenced_actual_data = current_data[:original_index[-1]]
            target = [np.expand_dims(sequenced_actual_data[i + sequence_length], 0) for i in
                      range(0, sequenced_actual_data.shape[0] - sequence_length-1, step_size)]
            actual_data = [np.expand_dims(sequenced_actual_data[i:i + sequence_length], 0) for i in
                                 range(0, sequenced_actual_data.shape[0] - sequence_length-1, step_size)]
            actual_data = np.concatenate(actual_data)
            sequenced_actual_datas.append(sequenced_actual_data)
            target = np.concatenate(target)
            actual_datas.append(actual_data)
            targets.append(target)

            actual_height = height[idx][:original_index[-1]]
            actual_height = [np.expand_dims(actual_height[i:i + sequence_length], 0) for i in
                           range(0, actual_height.shape[0] - sequence_length, step_size)]
            actual_height = np.concatenate(actual_height)
            actual_heights.append(actual_height)

            #TODO might need to fix this
        # there are 6 cases with different y's (height) of the plane
        # since we combined all the cases into one long array (to make it easier to fit the AI)
        # we want to keep track at which index is the next case
        # this shows at which index it goes to the next case
        # this will make it easy for us to plot the prediction later on
        self.next_case_index = [0]  # remember which one is the next case

        # this tells us all the valid index that we can get
        # this is because if we combine all the cases together, there are some indexes that will overflow from one case
        # into another and this will cause undefined distribution
        # we only want to interpolate between probes in a case and not combining one probe in one case to a different case
        self.all_valid_index = []
        for case_index, case in enumerate(actual_datas):
            next_index = self.next_case_index[-1] + case.shape[0]
            self.all_valid_index = self.all_valid_index + [i for i in range(self.next_case_index[-1], next_index-sparsity)]
            self.next_case_index.append(next_index)

        all_sequenced_x = np.concatenate(sequenced_interpolated_datas, 0)
        all_x = np.concatenate(interpolated_datas, 0)
        all_sequenced_actual_x = np.concatenate(sequenced_actual_datas, 0)
        all_actual_x = np.concatenate(actual_datas, 0)
        all_interpolated_targets = np.concatenate(interpolated_target_datas, 0)
        all_targets = np.concatenate(targets, 0)
        all_height = np.concatenate(height, 0)

        self.data = torch.tensor(all_x).float()
        self.sequenced_data = torch.tensor(all_sequenced_x).float()
        self.sequenced_actual_data = torch.tensor(all_sequenced_actual_x).float()
        self.actual_data = torch.tensor(all_actual_x).float()
        self.interpolated_target = torch.tensor(all_interpolated_targets).float()
        self.target = torch.tensor(all_targets).float()
        self.height = torch.tensor(all_height).float()
        self.sparsity = sparsity

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        target = self.target[idx]

        return x, target, self.height[idx]

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity


class TimeSeriesIntermediateSelfSupervisedDataset(Dataset):
    def __init__(self, data, height, step_size=2, sequence_length=3, sparsity=2, mask_value=0, mask_prob=0.3):
        """

                :param data:                the pressure signal of the location
                :param height:              the height of the probe coord
                :param sparsity:
                :param scaler:
                :param y_scaler:
                :param normalize:
                """
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.mask_prob = mask_prob
        # create interpolation model based on each case
        datas = []
        targets = []
        actual_heights = []
        self.interpolation_models = []
        for idx, current_data in enumerate(data):

            # TODO: HERE for self supervised learning
            # 1. Random masking
            # 2. Masking an entire segment (fixed)
            # 3. Mask segments (vaiable)
            # 4. Adding noise to data
            # 5.
            masked_data = self._random_masking(current_data, mask_value, self.mask_prob)
            processed_masked_data = np.array([masked_data[i:i+sequence_length]
                                              for i in range(0, masked_data.shape[0]-step_size-sequence_length, step_size)])
            processed_target_data = np.array([current_data[i:i+sequence_length]
                                              for i in range(0, current_data.shape[0]-step_size-sequence_length, step_size)])

            datas.append(processed_masked_data)
            targets.append(processed_target_data)


            # TODO might need to fix this
        # there are 6 cases with different y's (height) of the plane
        # since we combined all the cases into one long array (to make it easier to fit the AI)
        # we want to keep track at which index is the next case
        # this shows at which index it goes to the next case
        # this will make it easy for us to plot the prediction later on
        self.next_case_index = [0]  # remember which one is the next case

        # this tells us all the valid index that we can get
        # this is because if we combine all the cases together, there are some indexes that will overflow from one case
        # into another and this will cause undefined distribution
        # we only want to interpolate between probes in a case and not combining one probe in one case to a different case
        self.all_valid_index = []
        for case_index, case in enumerate(datas):
            next_index = self.next_case_index[-1] + case.shape[0]
            self.all_valid_index = self.all_valid_index + [i for i in
                                                           range(self.next_case_index[-1], next_index - sparsity)]
            self.next_case_index.append(next_index)

        all_x = np.concatenate(datas, 0)
        all_targets = np.concatenate(targets, 0)
        all_height = np.concatenate(height, 0)

        self.data = torch.tensor(all_x).float()
        self.target = torch.tensor(all_targets).float()
        self.height = torch.tensor(all_height).float()
        self.sparsity = sparsity

    def _random_masking(self, data, mask_value, prob=0.3):
        random_mask = np.random.random(data.shape) < prob
        masked_data = np.where(random_mask, mask_value, data)
        return masked_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        target = self.target[idx]

        return x, target, self.height[idx]

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity



class TimeSeriesForecastDataset(Dataset):
    def __init__(self, data, y, sequence_length=30, target_length=1, scaler=None, y_scaler=None, normalize=True):
        """

        :param data:                the pressure signal of the location
        :param y:                   y is not the target. y is the height of the probe coord
        :param sequence_length:
        :param scaler:
        :param y_scaler:
        :param normalize:
        """
        # normalize data
        current_data = data
        current_data = np.concatenate([i for i in current_data])
        current_y = np.concatenate([i for i in y])

        normalized_data = np.reshape(current_data, [-1, current_data.shape[1] * current_data.shape[2]])
        self.scaler = MinMaxScaler() if scaler is None else scaler
        normalized_data = self.scaler.fit_transform(normalized_data)
        normalized_data = np.reshape(normalized_data, [-1, current_data.shape[1], current_data.shape[2]])

        # normalize y
        normalize_y = np.reshape(current_y, [current_y.shape[0], -1])
        self.y_scaler = MinMaxScaler() if y_scaler is None else y_scaler
        normalize_y = self.y_scaler.fit_transform(normalize_y)
        y = np.reshape(normalize_y, [*current_y.shape])

        self.data = normalized_data[:-1]
        self.target = normalized_data[1:]
        self.y = torch.tensor(y).float()
        self.sequence_length = sequence_length
        self.target_length = target_length

        # there are 6 cases with different y's (height) of the plane
        # since we combined all the cases into one long array (to make it easier to fit the AI)
        # we want to keep track at which index is the next case
        # this shows at which index it goes to the next case
        # so we can skip the indexes that will combine different cases with each other
        # we don't want sequences from different cases to connect with each other otherwise
        # it will create undefined behaviour because last few sequence from a case should not
        # be combined with the first few sequence from a different case
        self.next_case_index = []
        total_iter = 0
        for current_data in data:
            self.next_case_index.append(total_iter + current_data.shape[0]-sequence_length-self.target_length-1)
            total_iter += current_data.shape[0]

    def __len__(self):
        return self.data.shape[0] - ( (self.sequence_length+self.target_length-1) * len(self.next_case_index))

    def __getitem__(self, idx):
        for next_case_idx in self.next_case_index:
            if idx > next_case_idx:
                idx += self.sequence_length
            else:
                break

        return np.reshape(self.data[idx:idx+self.sequence_length], [self.data[idx:idx+self.sequence_length].shape[0], -1]), \
            np.reshape(self.target[idx+self.sequence_length:idx+self.sequence_length+self.target_length],
                       [self.target[idx+self.sequence_length:idx+self.sequence_length+self.target_length].shape[0], -1]), \
        np.reshape(self.y[idx:idx+self.sequence_length], [self.y[idx:idx+self.sequence_length].shape[0], -1])


if __name__ == '__main__':
    # data contains [time, pressure]
    # current_y is the height of the current case
    data = load_csv(1, 1, case=1)
