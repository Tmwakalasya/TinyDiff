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


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output


class MLP:
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer.__call__(x)
        return x


xs = [[2.0, 3.0, 4.0],
      [3.0, 8.0, 5.0],
      [4.0, 6.0, 7.0],
      [7.0, 8.0, 9.0],
      ]
predictions = [Value(P) for P in [1.0,-1.0,-1.0,1.0]]
x = [2.0, 3.0, -1]
n = MLP(3, [4, 4, 1])
ypred = [n(x) for x in xs]
loss = [(yout - ygt) ** 2 for ygt, yout in zip(predictions,ypred)]
print(f"Loss: {loss}")
print(ypred)
output = n(x)
print(output)
