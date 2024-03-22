from magicml import Linear, Module
import numpy as np


class SimpleNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = np.maximum(x, 0)  # ReLU activation
        x = self.layer2(x)
        return x


# Example usage
if __name__ == '__main__':

    input_dim = 10
    hidden_dim = 5
    output_dim = 2

    model = SimpleNN(input_dim, hidden_dim, output_dim)
    input_data = np.random.randn(3, input_dim)  # batch size of 3
    output = model(input_data)
    print(output)
