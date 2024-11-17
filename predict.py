import numpy as np
import torch
import mlflow
import random
from pytorch_lightning.loggers import MLFlowLogger

from data import load_probe, load_csv, load_all_case
from trainer_type.interpolate import InterpolateType
from trainer_type.forecast import ForecastType
from data import load_probe, load_csv, load_all_case, TimeSeriesDataset, TimeSeriesIntermediateDataset, DotDict

#TODO change y to height because y is very ambiguous

# seeding to make reproducibility
random.seed(0)
torch.random.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)

process_dict = {
    'interpolate': InterpolateType,
    'forecast': ForecastType
}

process = 'interpolate'
is_forecast = True if process == 'forecast' else False


### Hyperparameters to change
process = 'interpolate'
is_forecast = True if process == 'forecast' else False
running_type = 'probe'
model_path = 'model/self_supervised/oneprobe1_1_case00.3.ckpt'
# self supervised
is_self_supervised = False
params = {
    'is_self_supervised': is_self_supervised,
    'process': 'interpolate',
    'sparsity': 5,
    'step_size': 3,
    'sequence_length': 3 * 2,
    'is_attention': False,
    'hidden_multiplier': 64,
    'latent_dim': 4 * 64,
    'num_layers': 1,
    'learning_rate': 0.002,
    'running_type': f'{"self_supervised_" if is_self_supervised else ""}{running_type}', #[probe, case, all, synthetic]
    'model_type': 'mamba',
}
config = DotDict(params)

process_str = f'forecast_{params["sequence_length"]}_{params["sparsity"]}' if process == 'forecast' else 'interpolate'
input_shape = [1, 1]

def synthetic_data():
    all_time = []
    synthetic_data = []
    for current_time in range(10000):
        time = current_time / 100
        value = np.sin(2 * np.pi * time) - np.cos(2 * np.pi * time/4) + np.cos(np.pi * time)
        synthetic_data.append(value)
        all_time.append(time)

    synthetic_data = np.array(synthetic_data)
    mean = np.mean(synthetic_data)
    normalized_synthetic_data = synthetic_data - mean
    abs_max = np.max(np.abs(normalized_synthetic_data))
    normalized_synthetic_data = normalized_synthetic_data / abs_max

    return [{'mean': mean, 'abs_max':abs_max, 'value': np.vstack([all_time, normalized_synthetic_data]).T}]

if running_type == 'probe':
    case = 0
    x_pos = 8
    z_pos = 1
    model_dirpath = f'lightning_logs/{config.model_type}_case{case}_sparsity{config.sparsity}_{config.step_size}_{config.sequence_length}_{x_pos}_{z_pos}'
    model_filename = f'case{case}_sparsity{config.sparsity}_{x_pos}_{z_pos}'
    x, y = load_probe(case, x_pos, z_pos, None)
    input_shape = [1, 1]
    data_filename = 'Data/1_1.npy'
    config.running_type += f'{x_pos}_{z_pos}_case{case}'
elif running_type == 'case':
    case = 7
    model_dirpath = f'lightning_logs/{config.model_type}_case{case}_sparsity{config.sparsity}_{config.step_size}_{config.sequence_length}'
    model_filename = f'case{case}_sparsity{config.sparsity}'
    x, y = load_csv(case=case) #abc = x[0]['value'].max() - x[0]['value'].min(); print(abc)
    input_shape = [11, 5]
    data_filename = f'Data/case_{case}.npy'
    config.running_type += f'case{case}'
elif running_type == 'all':
    model_dirpath = f'lightning_logs/{config.model_type}_all_sparsity{config.sparsity}_{config.step_size}_{config.sequence_length}'
    model_filename = f'all_sparsity{config.sparsity}'
    x, y = load_all_case()
    input_shape = [11, 5]
    data_filename = 'Data/all.npy'
    config.running_type += f'all_case'
elif running_type == 'synthetic':
    x = synthetic_data()
    y = np.copy(x)
    input_shape = [1, 1]
    data_filename = 'Data/synthetic.npy'
    config.running_type += f'synthetic'

total_shape = input_shape[0] * input_shape[1]
np.save(data_filename, [x, y])



# process_type = process_dict[process](running_type=f'trained_all_predict_one_case{case}')
process_type = process_dict[process](config.running_type, config.model_type)
# process_type = process_dict[process](running_type=f'pre_SBLI_{running_type}')
x_train, x_test, y_train, y_test = process_type.create_train_test_split(x, y, test_size=0.2, shuffle=False)
train_dataloader, test_dataloader = process_type.generate_dataloader(x_train, x_test, y_train, y_test, config.step_size, config.sequence_length, config.sparsity)
model = process_type.generate_model(total_shape, config.latent_dim, total_shape, config.num_layers, config.is_attention, config.hidden_multiplier, config.sequence_length, config.sparsity, config.learning_rate, model_type=config.model_type)
# model.load_state_dict(torch.load(model_path)['state_dict'])

experiment_name = 'main'
run_name = f'{config.model_type}_{config.running_type}'
# mlflow.set_experiment(experiment_name)
# mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, log_model=False)
# mlflow.pytorch.autolog()

# r2 = 1 - (rmse**2) / np.std(process_type.test_dataset.target.numpy()) ** 2
process_type.plot_prediction(model, x[0]['abs_max'], dataset_type='test_dataset', input_shape=input_shape)



