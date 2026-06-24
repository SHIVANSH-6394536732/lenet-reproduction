import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import LeNet5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load trained model
model = LeNet5().to(device)
model.load_state_dict(torch.load('lenet5_mnist.pth'))
model.eval()

# Collect all predictions
all_preds = []
all_labels = []
misclassified_images = []
misclassified_preds = []
misclassified_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        wrong_mask = predictions != labels
        if wrong_mask.sum() > 0 and len(misclassified_images) < 10:
            misclassified_images.extend(images[wrong_mask].cpu().numpy())
            misclassified_preds.extend(predictions[wrong_mask].cpu().numpy())
            misclassified_labels.extend(labels[wrong_mask].cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - LeNet5 on MNIST')
plt.savefig('../results/confusion_matrix.png')
print("Saved confusion_matrix.png")

# Misclassified Examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    if idx < len(misclassified_images):
        ax.imshow(misclassified_images[idx][0], cmap='gray')
        ax.set_title(f"True: {misclassified_labels[idx]}, Pred: {misclassified_preds[idx]}")
        ax.axis('off')
plt.tight_layout()
plt.savefig('../results/misclassified_examples.png')
print("Saved misclassified_examples.png")

plt.show()
