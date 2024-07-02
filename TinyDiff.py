import math


class Value:
    def __init__(self, data, initial_value=set(), operation='', label=''):
        self.data = data
        self.previous_vals = set(initial_value)
        self.operation = operation
        self.label = label
        self.gradient = 0.0
        self._backward = lambda: None

    def __repr__(self):
        """This provides a small string representation"""
        output = f"Value= {self.data}"
        return output

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.gradient += 1.0 * output.gradient
            other.gradient += 1.0 * output.gradient

        output._backward = _backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient

        output._backward = _backward
        return output

    def __rmul__(self, other):
        # Use if __mul__ won't work:
        return self * other

    def __sub__(self, other):
        output = Value(self.data - other.data, (self, other), operation="-")

        def _backward():
            self.gradient += 1.0 * output.gradient
            other.gradient += -1.0 * output.gradient

        output._backward = _backward
        return output

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float)), "Power must be int or float"
        exponent = Value(self.data ** power, (self,), "**")

        def _backward():
            self.gradient += (power * self.data ** (power - 1)) * exponent.gradient

        exponent._backward = _backward
        return exponent

    def __truediv__(self, other):
        if other.data == 0:
            raise ValueError("Division by Zero")
        output = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.gradient += (1 / other.data) * output.gradient
            other.gradient += (-self.data / (other.data ** 2)) * output.gradient

        output._backward = _backward
        return output

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(t, (self,), 'tanh')

        def _backward():
            self.gradient += (1 - t ** 2) * output.gradient

        output._backward = _backward
        return output

    def sigmoid(self):
        x = self.data
        if x > 709:  # math.log(sys.float_info.max)
            s = 1.0
        elif x < -709:
            s = 0.0
        elif x >= 0:
            z = math.exp(-x)
            s = 1 / (1 + z)
        else:
            z = math.exp(x)
            s = z / (1 + z)
        output = Value(s, (self,), 'sigmoid')

        def _backward():
            self.gradient += s * (1 - s) * output.gradient

        output._backward = _backward
        return output

    def relu(self):
        output = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.gradient += (self.data > 0) * output.gradient

        output._backward = _backward
        return output

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous_vals:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        print(topo)

        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x1w1.label = 'x1*w1'
x2w2 = x2 * w2
x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b
n.label = 'n'
o = n.tanh()
o.label = 'o'

# Compute gradients
o.backward()

# Print results
print(f"o.gradient: {o.gradient}")
print(f"n.gradient: {n.gradient}")
print(f"b.gradient: {b.gradient}")
print(f"x1w1x2w2.gradient: {x1w1x2w2.gradient}")
print(f"x1.gradient: {x1.gradient}")
print(f"x2.gradient: {x2.gradient}")
print(f"w1.gradient: {w1.gradient}")
print(f"w2.gradient: {w2.gradient}")
A = Value(3)
z = A + 1
print(z)
