__all__ = ["MLP"]
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with user-specified parameters.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers: int = 3,
        nonlinearity=nn.Tanh,
    ):
        super(MLP, self).__init__()

        # we require at least 2 layers
        assert num_layers >= 2

        layers = [nn.Linear(input_dim, hidden_dim), nonlinearity()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nonlinearity()]
        layers += [nn.Linear(hidden_dim, output_dim)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
