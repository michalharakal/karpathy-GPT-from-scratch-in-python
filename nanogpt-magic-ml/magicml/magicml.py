import numpy as np


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self.parameters = []

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def parameters(self):
        """Return a list of parameters for the module."""
        return self.parameters

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)


class Linear(Module):
    """Applies a linear transformation to the incoming data."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.bias = np.zeros(out_features)
        self.parameters += [self.weight, self.bias]

    def forward(self, x):
        return np.dot(x, self.weight.T) + self.bias


class Embedding(Module):
    """A simple embedding layer."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = np.random.randn(num_embeddings, embedding_dim) * 0.01
        self.parameters += [self.embeddings]

    def forward(self, indices):
        return self.embeddings[indices]


# A very simplistic optimizer for demonstration purposes
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param)
