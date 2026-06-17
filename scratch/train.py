import numpy as np
from scratch.network import LeNet5


fake_image = np.random.randn(28,28)

model = LeNet5()

output = model.forward(fake_image)

print("Output Shape:" , output.shape)
print("Output Values: " , output)

