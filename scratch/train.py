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


# Split into train/test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

model = LeNet5()
loss_fn = CrossEntropyLoss()

num_samples = 1000  # small subset first, to test quickly
learning_rate = 0.0008
for epoch in range(3):
    total_loss = 0
    correct = 0

    for idx in range(num_samples):
        image = X_train[idx]
        label = y_train[idx]

        # Forward pass
        raw_output = model.forward(image)
        probs = softmax(raw_output)

        # Loss
        loss = loss_fn.forward(probs, label)
        total_loss += loss


        if loss > 15:
            print(f"WARNING: Sample {idx} hit epsilon floor (loss={loss:.4f}). Prediction collapsed.")

        if np.isnan(loss):
            print(f"NaN detected at sample {idx}! Raw output was:", raw_output)
            break

        prediction = np.argmax(probs)
        if prediction == label:
            correct += 1

        # Backward pass
        # Backward pass
        grad = loss_fn.backward()
        grad = np.clip(grad, -1, 1)  
        grad = model.fc3.backward(grad, learning_rate)
        grad = np.clip(grad, -1, 1)
        grad = model.relu4.backward(grad)
        grad = model.fc2.backward(grad, learning_rate)
        grad = np.clip(grad, -1, 1)
        grad = model.relu3.backward(grad)
        grad = model.fc1.backward(grad, learning_rate)
        grad = np.clip(grad, -1, 1)
        grad = model.pool2.backward(grad)
        grad = model.relu2.backward(grad)
        grad = model.conv2.backward(grad, learning_rate)
        grad = np.clip(grad, -1, 1)
        grad = model.pool1.backward(grad)
        grad = model.relu1.backward(grad)
        grad = model.conv1.backward(grad, learning_rate)

        if idx % 20 == 0:
            print(f"Sample {idx}, Loss: {loss:.4f}, Correct so far: {correct}/{idx+1}")

    print(f"\nEpoch {epoch+1} done. Avg Loss: {total_loss/num_samples:.4f}, Accuracy: {correct}/{num_samples}")