import numpy as np

class Conv2D:
    def __init__(self, num_filters , filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(
            num_filters , filter_size , filter_size
        ) * 0.1
    
    def iterate_regions(self,image):
        h,w = image.shape
        for i in range (h - self.filter_size - 1):
            for j in range ( w - self.filter_size - 1 ):
                patch =  image[i: i + self.filter_size, j: j + self.filter_size]
                yield patch , i , j

    def forward(self,image):
        self.last_input = image
        h,w = image.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w , self.num_filters))


        for patch , i , j in self.iterate_regions(image):
            output[i , j] = np.sum(patch  * self.filters, axis = (1,2))

        return output
