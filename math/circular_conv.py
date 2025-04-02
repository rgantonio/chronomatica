import numpy as np

# Simple circular convolution implementation
def circular_convolution(x, h):

    # Get length
    vec_len = len(x)

    # Initialize
    y = np.zeros(vec_len)

    # For loop for circular convolution
    # Observe that it isn't a regular circular convolution
    for i in range(vec_len):
        for j in range(vec_len):
            y[i] += x[j] * h[(i - j) % vec_len]
    
    return y

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    h = np.array([6, 7, 8, 9, 10])
    y = circular_convolution(x, h)
    print(y)