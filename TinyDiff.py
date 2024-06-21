class Value:
    def __init__(self, data, initial_value=set(), operation='', label=''):
        self.data = data
        self.previous_vals = set(initial_value)
        self.operation = operation
        self.label = label
        self.gradient = 0.0

    def __repr__(self):
        """This provides a small string representation"""
        output = f"Value= {self.data}"
        return output

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), "+")
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), "*")
        return output

    def __sub__(self, other):
        output = Value(self.data - other.data, (self, other), operation="-")
        return output

    def __pow__(self, power, modulo=None):
        exponent = Value(self.data ** power.data, (self, power), "**")
        return exponent

    def __truediv__(self, other):
        if other.data == 0:
            raise ValueError("Division by Zero")
        output = Value(self.data / other.data, (self, other), "/")
        return output
