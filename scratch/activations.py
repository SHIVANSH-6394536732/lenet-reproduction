import numpy as np

class ReLu:
    def forward(self,x): 
        self.last_input = x
        return np.maximum(0,x)
    
    def backward(self , grad_output):
        return grad_output * (self.last_input > 0)


class Tanh:
    def forward(self,x):
        self.last_input = x
        return np.tanh(x)


    def backward(self, grad_output):
        return grad_output * (1 -np.tanh(self.last_input) ** 2)

class Sigmoid:
    def forward(self,x):
        self.last_input = x
        self.last_output = 1 / (1 + np.exp(-x))
        return self.last_output

    def backward(self,grad_output):
        return grad_output * self.last_output * (1-self.last_output)