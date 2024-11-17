import mlflow
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import load_probe, load_csv, load_all_case, DotDict
from trainer_type.interpolate import SelfSupervisedInterpolatedType

params = {
    'process': 'interpolate',
    'sparsity': 3,
    'mask_prob': 0.3,
    'step_size': 3 * 2,
    'sequence_length': 3 * 2,
    'is_attention': False,
    'hidden_multiplier': 64,
    'latent_dim': 4 * 64,
    'num_layers': 1,
    'learning_rate': 0.00002,
    'running_type': 'case' #[probe, case, all, synthetic]
}
config = DotDict(params)

# data ingestion
if config.running_type == 'probe':
    case = 0
    x_pos = 1
    z_pos = 1
    x, y = load_probe(case, x_pos, z_pos, None)
    input_shape = [1, 1]
    data_filename = 'Data/1_1.npy'
    config.running_type = f'oneprobe{x_pos}_{z_pos}_case{case}'
elif config.running_type == 'case':
    case = 0
    input_shape = [9, 1] if case == 0 else [11, 5]
    x, y = load_csv(case=case, input_shape=(9, 1))
    data_filename = f'Data/case_{case}.npy'
    config.running_type = f'case{case}'
elif config.running_type == 'all':
    x, y = load_all_case()
    input_shape = [11, 5]
    data_filename = 'Data/all.npy'
    config.running_type = f'all_case'
total_shape = input_shape[0] * input_shape[1]

# mlflow declaration
dataset = mlflow.data.from_numpy(np.array([x, y]))
experiment_name = 'self_supervised'
run_name = config.running_type
mlflow.set_experiment(experiment_name)
mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, log_model=False)

with mlflow.start_run(run_id=mlflow_logger.run_id):
    mlflow.log_input(dataset)
    mlflow.log_params(params)
    mlflow.pytorch.autolog()

    # data processing
    interpolate_process = SelfSupervisedInterpolatedType(config.running_type)
    x_train, x_test, y_train, y_test = interpolate_process.create_train_test_split(x, y, test_size=0.1, shuffle=False)
    train_loader, test_loader = interpolate_process.generate_dataloader(x_train, x_test, y_train, y_test,
                                            config.step_size, config.sequence_length, config.sparsity, config.mask_prob)

    # model declaration
    model = interpolate_process.generate_model(total_shape,
                                               config.latent_dim, total_shape,
                                               config.num_layers, config.is_attention, config.hidden_multiplier,
                                               config.sequence_length, config.sparsity, config.learning_rate, return_all=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=os.path.join('model/self_supervised'),
                                          filename=config.running_type+str(config.mask_prob))


    # model training
    trainer = pl.Trainer(max_epochs=50,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_dataloaders=test_loader)
    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])

    # model prediction
    interpolate_process.plot_prediction(model, x[0]['abs_max'], dataset_type='train_dataset', input_shape=input_shape)
    interpolate_process.plot_prediction(model, x[0]['abs_max'], dataset_type='test_dataset', input_shape=input_shape)
