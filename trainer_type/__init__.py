from abc import abstractmethod


class TrainerType:
    def __init__(self):
        self._model = None
        self._dataloader = None
        self._train_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._test_dataloader = None

    @property
    def model(self):
        return self._model

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @abstractmethod
    def generate_model(self, input_dim, latent_dim, output_dim, num_layers, is_attention, hidden_multiplier, sequence_length, sparsity, learning_rate, return_all, model_type):
        ...

    @abstractmethod
    def create_train_test_split(self, *arrays, test_size=0.2, shuffle=False):
        ...

    @abstractmethod
    def generate_dataloader(self, x_train, x_test, y_train, y_test, target_length):
        ...

    @abstractmethod
    def plot_prediction(self, model, wandb_logger):
        ...
