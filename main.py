import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import random

from data import load_probe, load_csv, load_all_case, TimeSeriesDataset, TimeSeriesIntermediateDataset, DotDict

from trainer_type.interpolate import InterpolateType
from trainer_type.forecast import ForecastType

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

### Hyperparameters to change
process = 'interpolate'
is_forecast = True if process == 'forecast' else False
running_type = 'all'
# self supervised
is_self_supervised = False
model_path = 'model/self_supervised/oneprobe1_1_case00.3-v1.ckpt'

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
    'model_type': 'mamba',  # [transformer, mamba]
}
config = DotDict(params)

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
    std = np.std(synthetic_data)
    normalized_synthetic_data = synthetic_data - mean
    abs_max = np.max(np.abs(normalized_synthetic_data))
    normalized_synthetic_data = normalized_synthetic_data / std

    return [{'mean': mean, 'abs_max':abs_max, 'value': np.vstack([all_time, normalized_synthetic_data]).T}]


if running_type == 'probe':
    case = 6
    x_pos = 5
    z_pos = 3
    model_dirpath = f'lightning_logs/{config.model_type}_case{case}_sparsity{config.sparsity}_{config.step_size}_{config.sequence_length}_{x_pos}_{z_pos}'
    model_filename = f'case{case}_sparsity{config.sparsity}_{x_pos}_{z_pos}'
    x, y = load_probe(case, x_pos, z_pos, None)
    input_shape = [1, 1]
    data_filename = 'Data/1_1.npy'
    config.running_type += f'{x_pos}_{z_pos}_case{case}'
elif running_type == 'case':
    case = 6
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

# process data
process_type = process_dict[process](config.running_type, config.model_type)
x_train, x_test, y_train, y_test = process_type.create_train_test_split(x, y, test_size=0.2, shuffle=False)
train_dataloader, test_dataloader = process_type.generate_dataloader(x_train, x_test, y_train, y_test, config.step_size, config.sequence_length, config.sparsity)
model = process_type.generate_model(total_shape, config.latent_dim, total_shape, config.num_layers, config.is_attention, config.hidden_multiplier, config.sequence_length, config.sparsity, config.learning_rate, model_type=config.model_type)

# load self supervised model if exists
if is_self_supervised:
    loaded_state_dict = torch.load(model_path)['state_dict']
    filtered_state_dict = {k: v if v.size() == model.state_dict()[k].size() else model.state_dict()[k]
                           for k, v in loaded_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict)

# create model
experiment_name = 'main'
run_name = f'{config.model_type}_{config.running_type}'
mlflow.set_experiment(experiment_name)
mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, log_model=False)
mlflow.pytorch.autolog()
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=model_dirpath, filename=model_filename)

# start training
with mlflow.start_run(run_id=mlflow_logger.run_id):
    dataset = mlflow.data.from_numpy(np.array([x, y]))
    mlflow.log_input(dataset)
    mlflow.log_params(params)
    mlflow.pytorch.autolog()
    trainer = pl.Trainer(max_epochs=50,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader)
    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])

    process_type.plot_prediction(model, x[0]['abs_max'], dataset_type='train_dataset', input_shape=input_shape)
    process_type.plot_prediction(model, x[0]['abs_max'], dataset_type='test_dataset', input_shape=input_shape)

