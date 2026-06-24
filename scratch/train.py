import numpy as np
from scratch.network import LeNet5
from scratch.loss import softmax, CrossEntropyLoss

def load_mnist():
    from sklearn.datasets import fetch_openml
    print("Downloading MNIST (only happens once, then cached)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    X = X.reshape(-1, 28, 28)

    return X, y

X, y = load_mnist()
print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("First label:", y[0])