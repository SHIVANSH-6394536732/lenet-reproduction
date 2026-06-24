import numpy as np 

def softmax(x):
    shifted = x - np.max(x)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)

class CrossEntropyLoss:
    def forward(self, predictions, true_label):
        self.true_label = true_label

        epsilon = 1e-7
        self.predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.log(self.predictions[true_label])
        return loss

    def backward(self):
        grad = np.zeros_like(self.predictions)
        grad[self.true_label] = -1 / self.predictions[self.true_label]
        return grad