import numpy as np

image = np.array([
    [1, 3, 2, 1],
    [5, 6, 1, 2],
    [7, 2, 4, 3],
    [1, 8, 2, 9]
])

def max_pooling(image):
    poolsize = 2
    stride = 2
    h, w = image.shape

    output = np.zeros(
        (
            (h - poolsize) // stride + 1,
            (w - poolsize) // stride + 1
        )
    )

    for i in range(0, h - poolsize + 1, stride):
        for j in range(0, w - poolsize + 1, stride):
            output[i // stride][j // stride] = np.max(
                image[i:i + poolsize, j:j + poolsize]
            )   

    return output

def flatten(image):
    h, w = image.shape
    result = np.zeros(h * w)
    for i in range(h):
        for j in range(w):
            result[i * w + j] = image[i][j]
    return result

max_pooling_result = max_pooling(image)
print("max_pooling_result:")
print(max_pooling_result)

flatten_result = flatten(max_pooling_result)
print("flatten_result:")
print(flatten_result)