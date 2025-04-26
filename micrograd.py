import numpy as np
import math

class Value:

    def __init__(self, data, label= None, children= tuple()):
        self.data = data
        self._backward = lambda: None
        self._children = set(children)
        self.grad = 0.00
        self.label = label
    
    def __repr__(self):
        return f"Value(data= {self.data}, label= {self.label})"

    def __add__(self, other):

        if not isinstance(other, Value): other = Value(other) 
        out= Value((self.data + other.data), children= (self, other))

        def _backward():
            self.grad += 1* out.grad
            other.grad += 1* out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):

        if not isinstance(other, Value): other = Value(other) 
        out= Value((self.data - other.data), children= (self, other))

        def _backward():
            self.grad += 1* out.grad
            other.grad += -1* out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):

        if not isinstance(other, Value): other = Value(other) 
        out = Value((self.data * other.data), children= (self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):

        if not isinstance(other, Value): other = Value(other) 
        out = Value((self.data / other.data), children= (self, other))

        def _backward():
            self.grad += out.grad * (1/other.data)
            other.grad += out.grad * (-self.data/(other.data**2))

        out._backward = _backward
        return out
    
    def exp(self):

        out = Value(math.exp(self.data), children= (self,))

        def _backward():
            self.grad += out.grad * (math.exp(self.data))

        out._backward = _backward
        return out
     
    def tanh(self):

        x = self.data
        t = (np.exp(2*x) - 1)/ (np.exp(2*x) + 1)
        out= Value(t, children= (self,))

        def _backward():
            self.grad += out.grad*(1 - t**2)
        out._backward = _backward

        return out
    
    def __pow__(self, power):

        out = Value(self.data**power, children=(self,))

        def _backward():
            self.grad += power*(self.data**(power-1))* out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        result = list()
        visited = set()

        def topo(head):
            if head not in visited:
                visited.add(head)
                for child in head._children:
                    topo(child)
                result.append(head)  # append AFTER children for correct topological sort

        topo(self)

        self.grad = 1.0 
        for node in reversed(result):
            node._backward()  

class Neuron:

    def __init__(self, nin):
        self.W = [Value(np.random.normal(size= (1,)).item()) for _ in range(nin)]
        self.b = Value(np.random.normal(size= (1,)).item())

    def __call__(self, x):
        out = sum(((input*weight) for input, weight in zip(x, self.W)),self.b)
        return out.tanh()

    def parameters(self):
        return self.W + [self.b]

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        params = list()
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        
        return params

class MLP:

    def __init__(self, nin, nouts):
        shapes = [nin] + nouts
        self.layers = [Layer(shapes[i], shapes[i+1]) for i in range(len(nouts))]
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        
        if len(x) == 1: return x[0]
        return x

    def parameters(self):

        params = list()
        for layer in self.layers:
            params.extend(layer.parameters())
        
        return params


x = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]
y_true = [1.0, -1.0, -1.0, 1.0]

#instantiate the model
mlp = MLP(3,[4,4,1])

# hyperparameters
learning_rate = 0.05
epochs = 100

#full training loop
for i in range(epochs):

    # reset gradients
    for parameter in mlp.parameters():
        parameter.grad = 0
    
    # forward pass
    y_pred = [mlp(obs) for obs in x]

    # compute loss
    loss = sum(((yt - yp)**2 for yt, yp in zip(y_true, y_pred)))
    print(f"Epoch: {i} - Loss: {loss.data:.3f}")

    # backward pass
    loss.backward()

    # update parameters with gradients
    for parameter in mlp.parameters():
        parameter.data -= (learning_rate*parameter.grad)

print(y_pred)