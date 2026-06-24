from warnings import simplefilter
import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size, input_channels=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.filters = np.random.randn(
            num_filters, filter_size, filter_size, input_channels
        ) * 0.1

    def iterate_regions(self, image):
        h, w, c = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                patch = image[i:i+self.filter_size, j:j+self.filter_size, :]
                yield patch, i, j

    def forward(self, image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        self.last_input = image
        h, w, c = image.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w, self.num_filters))

        for patch, i, j in self.iterate_regions(image):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(patch * self.filters[f])

        return output



    def backward(self, grad_output, learning_rate=0.1):
        grad_filters = np.zeros(self.filters.shape)
        grad_input = np.zeros(self.last_input.shape)

        for patch, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                grad_filters[f] += patch * grad_output[i, j, f]
                grad_input[i:i+self.filter_size, j:j+self.filter_size, :] += (
                    self.filters[f] * grad_output[i, j, f]
                )

        self.filters -= learning_rate * grad_filters

        return grad_input


class MaxPool2D:
    def __init__(self, pool_size = 2):
        self.pool_size = pool_size

    def forward(self,input):
        self.last_input = input
        h,w,num_filters = input.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size


        output = np.zeros((new_h , new_w , num_filters))


        for i in range(new_h):
            for j in range(new_w):
                region = input[
                    i*self.pool_size :   (i + 1) * self.pool_size,
                    j*self.pool_size :   (j + 1) * self.pool_size,
                    :

                ]

                output[i , j] = np.max(region, axis =(0,1))

        return output

    def backward(self , grad_output):
        grad_input = np.zeros(self.last_input.shape)
        h, w, num_filters = grad_output.shape

        for i in range(h):
            for j in range(w):
                region = self.last_input[
                    i*self.pool_size: (i+1) * self.pool_size,
                    j*self.pool_size: (j+1) * self.pool_size,
                    :
                ]
                for f in range(num_filters):
                    max_val = np.max(region[: , : , f])
                    for x in range(region.shape[0]):
                        for y in range(region.shape[1]):
                            if region[x,y,f] == max_val:
                                grad_input[
                                    i*self.pool_size +x,
                                    j*self.pool_size +y,
                                    f
                                ] = grad_output[i,j,f]
        return grad_input
 





class FullyConnected:
    def __init__(self,input_size , output_size):
        self.weights = np.random.randn(input_size , output_size) * 0.1
        self.bias  = np.zeros(output_size)


    def forward(self,input):
        self.last_input_shape  = input.shape
        self.last_input = input.flatten()
        output = np.dot(self.last_input , self.weights) + self.bias
        return output

    def backward(self, grad_output ,  learning_rate = 0.1):
        grad_weights = np.outer(self.last_input, grad_output)
        grad_bias = grad_output
        grad_input = np.dot(grad_output, self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input.reshape(self.last_input_shape)