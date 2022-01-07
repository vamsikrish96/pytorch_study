import random
import numpy as np
import torch
from typing import Tuple, Callable, List, NamedTuple
import torchvision
import tqdm

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    paddings: Tuple[int, ...] = (0, 0, 0)
    dense_hiddens: Tuple[int, ...] = (256, 256)


# Pytorch preliminaries
def gradient_norm(function: Callable, *tensor_list: List[torch.Tensor]) -> float:
    # TODO WRITE CODE HERE
    function(*tensor_list).backward()
    return torch.norm(torch.stack([t.grad for t in tensor_list])).item()

def jacobian_norm(function: Callable, input_tensor: torch.Tensor) -> float:
    # TODO WRITE CODE HERE
    jacobian = torch.autograd.functional.jacobian(function, input_tensor)
    return torch.linalg.norm(jacobian).item()

class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 datapath: str = './data',
                 n_classes: int = 10,
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 activation_name: str = "relu",
                 normalization: bool = True):
        self.train, self.valid, self.test = self.load_dataset(datapath)
        if normalization:
            self.train, self.valid, self.test = self.normalize(self.train, self.valid, self.test)
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0][0].shape
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], net_config,
                                           n_classes, activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], net_config, n_classes, activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = 1e-9

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': [],
                           'train_gradient_norm': []}

    @staticmethod
    def load_dataset(datapath: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        trainset = torchvision.datasets.FashionMNIST(root=datapath,
                                                     download=True, train=True)
        testset = torchvision.datasets.FashionMNIST(root=datapath,
                                                    download=True, train=False)

        X_train = trainset.data.view(-1, 1, 28, 28).float()
        y_train = trainset.targets

        X_ = testset.data.view(-1, 1, 28, 28).float()
        y_ = testset.targets

        X_val = X_[:2000]
        y_val = y_[:2000]

        X_test = X_[2000:]
        y_test = y_[2000:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        # TODO write code here
        temp_list = [input_dim] + list(net_config.dense_hiddens) + [n_classes]
        layers = []
        layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
        for i in range(len(temp_list)-1):
            if(i == len(temp_list)-2):
                layers.append(torch.nn.Linear(temp_list[i], temp_list[i+1]))
                break
            layers.append(torch.nn.Linear(temp_list[i], temp_list[i+1]))
            layers.append(activation)

            
        layers.append(torch.nn.Softmax(dim=1))
        return torch.nn.Sequential(*layers)
            


    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # TODO write code here
        cnn_list = [in_channels] + list(net_config.n_channels)
        temp_list = [net_config.n_channels[-1]*4*4] + list(net_config.dense_hiddens) + [n_classes]
        layers = []
        
        for i in range(len(cnn_list)-1):
            if(i == len(cnn_list)-2):
                layers.append(torch.nn.Conv2d(cnn_list[i], cnn_list[i+1], kernel_size=net_config.kernel_sizes[i], stride=net_config.strides[i], padding=net_config.paddings[i]))
                layers.append(activation)
                break
            layers.append(torch.nn.Conv2d(cnn_list[i], cnn_list[i+1], kernel_size=net_config.kernel_sizes[i], stride=net_config.strides[i], padding=net_config.paddings[i]))
            layers.append(activation)
            layers.append(torch.nn.MaxPool2d(kernel_size=2))

        layers.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
        layers.append(torch.nn.Flatten())
        for i in range(len(temp_list)-1):
            if(i == len(temp_list)-2):
                layers.append(torch.nn.Linear(temp_list[i], temp_list[i+1]))
                break
            layers.append(torch.nn.Linear(temp_list[i], temp_list[i+1]))
            layers.append(activation)

        layers.append(torch.nn.Softmax(dim=1))
        return torch.nn.Sequential(*layers)



    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        # TODO WRITE CODE HERE
        if(activation_str == "relu"):
            return torch.nn.ReLU()
        elif(activation_str == "sigmoid"):
            return torch.nn.Sigmoid()
        else:
            return torch.nn.Tanh()
            
                


    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        # TODO WRITE CODE HERE
        y_one_hot = torch.zeros(y.shape[0], self.n_classes)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        return y_one_hot

    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # TODO WRITE CODE HERE
        #Cross entropy loss and accuracy calculation
        y_pred = self.network(X)
        y_pred = torch.clip(y_pred, min=self.epsilon, max=(1 - self.epsilon))
        y_pred = torch.log(y_pred)
        y = torch.argmax(y, dim=1)
        loss = torch.nn.NLLLoss()
        loss_value = loss(y_pred, y)
        accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y).item() / y.shape[0]
        return loss_value, accuracy
        

    @staticmethod
    def compute_gradient_norm(network: torch.nn.Module) -> float:
        # TODO WRITE CODE HERE
        # Compute the Euclidean norm of the gradients of the parameters of the network
        # with respect to the loss function.
        total_norm= 0
        for p in network.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        # TODO WRITE CODE HERE
        self.optimizer.zero_grad()   # zero the gradient buffers
        output = self.network(X_batch)

        y_pred = torch.clip(output, min=self.epsilon, max=(1 - self.epsilon))
        y_pred = torch.log(y_pred)
        
        y = torch.argmax(y_batch, dim=1)
        loss = torch.nn.NLLLoss()
        loss_value = loss(y_pred, y)
        loss_value.backward()
        self.optimizer.step()
        #calculate gradient norm
        total_norm= 0
        for p in self.network.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
            
        return total_norm
        


    def log_metrics(self, X_train: torch.Tensor, y_train_oh: torch.Tensor,
                    X_valid: torch.Tensor, y_valid_oh: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train_oh)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid_oh)

        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(float(train_loss))
        self.train_logs['validation_loss'].append(float(valid_loss))

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        y_train_oh = self.one_hot(y_train)
        X_valid, y_valid = self.valid
        y_valid_oh = self.one_hot(y_valid)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train_oh[self.batch_size * batch:self.batch_size * (batch + 1), :]
                gradient_norm = self.training_step(minibatchX, minibatchY)
            # Just log the last gradient norm
            self.train_logs['train_gradient_norm'].append(gradient_norm)
            self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # TODO WRITE CODE HERE
        with torch.no_grad():
            y = self.one_hot(y)
            loss,accuracy = self.compute_loss_and_accuracy(X,y)
            return loss,accuracy

    @staticmethod
    def normalize(train: Tuple[torch.Tensor, torch.Tensor],
                  valid: Tuple[torch.Tensor, torch.Tensor],
                  test: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor]]:
        X_train, y_train = train
        X_valid, y_valid = valid
        X_test, y_test = test
        X_train_mean = X_train.mean(dim=0)
        X_train_std = X_train.std(dim=0)
        X_train_norm = (X_train - X_train_mean) / X_train_std
        X_valid_norm = (X_valid - X_train_mean) / X_train_std
        X_test_norm = (X_test - X_train_mean) / X_train_std
        return (X_train_norm, y_train), (X_valid_norm, y_valid), (X_test_norm, y_test)



    def test_equivariance(self):
        from functools import partial
        test_im = self.train[0][0]/255.
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(torchvision.transforms.functional.affine, angle=0,
                        translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(torchvision.transforms.functional.affine, angle=90,
                           translate=(0, 0), scale=1, shear=0)

        # TODO CODE HERE
        pass


