import random
from TinyDiff import Value


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Convert input x to Value objects if they aren't already
        x = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]

        # Use a Value object to accumulate the sum
        activation = Value(0)
        for wi, xi in zip(self.weights, x):
            activation = activation + wi * xi

        activation = activation + self.bias
        out = activation.tanh()
        return out


x = [2.0, 3.0]
n = Neuron(2)
output = n(x)
print(output)



