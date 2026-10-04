import numpy as np
from tensorflow.keras.datasets import mnist


# ==================================================
# Conv Forward
# ==================================================

def conv2d(image, kernel):

    h, w = image.shape

    kh, kw = kernel.shape

    output = np.zeros(
        (
            h - kh + 1,
            w - kw + 1
        )
    )

    for i in range(h - kh + 1):

        for j in range(w - kw + 1):

            total = 0

            for x in range(kh):

                for y in range(kw):

                    total += (
                        image[i + x][j + y]
                        * kernel[x][y]
                    )

            output[i][j] = total

    return output


# ==================================================
# ReLU
# ==================================================

def relu(x):

    return np.maximum(0, x)


# ==================================================
# Max Pool Forward
# ==================================================

def max_pool(image):

    pool_size = 2
    stride = 2

    height, width = image.shape

    output_height = (
        height - pool_size
    ) // stride + 1

    output_width = (
        width - pool_size
    ) // stride + 1

    output = np.zeros(
        (output_height, output_width)
    )

    max_positions = np.zeros(
        (output_height, output_width, 2),
        dtype=int
    )

    for i in range(
        0,
        height - pool_size + 1,
        stride
    ):

        for j in range(
            0,
            width - pool_size + 1,
            stride
    ):

            max_value = image[i][j]

            max_x = i
            max_y = j

            for x in range(pool_size):

                for y in range(pool_size):

                    value = image[i + x][j + y]

                    if value > max_value:

                        max_value = value

                        max_x = i + x
                        max_y = j + y

            output[i // stride][j // stride] = max_value

            max_positions[i // stride][j // stride] = [
                max_x,
                max_y
            ]

    return output, max_positions


# ==================================================
# Flatten
# ==================================================

def flatten(image):

    h, w = image.shape

    result = np.zeros(h * w)

    for i in range(h):

        for j in range(w):

            result[i * w + j] = image[i][j]

    return result


# ==================================================
# Max Pool Backward
# ==================================================

def max_pool_backward(
    d_output,
    max_positions,
    input_shape
):

    d_input = np.zeros(input_shape)

    output_height, output_width = d_output.shape

    for i in range(output_height):

        for j in range(output_width):

            max_x = max_positions[i][j][0]

            max_y = max_positions[i][j][1]

            d_input[max_x][max_y] += d_output[i][j]

    return d_input


# ==================================================
# Flatten Backward
# ==================================================

def flatten_backward(
    d_flat,
    original_shape
):

    height, width = original_shape

    d_input = np.zeros(
        (height, width)
    )

    index = 0

    for i in range(height):

        for j in range(width):

            d_input[i][j] = d_flat[index]

            index += 1

    return d_input


# ==================================================
# ReLU Backward
# ==================================================

def relu_backward(
    d_output,
    original_input
):

    height, width = original_input.shape

    d_input = np.zeros(
        (height, width)
    )

    for i in range(height):

        for j in range(width):

            if original_input[i][j] > 0:

                d_input[i][j] = d_output[i][j]

            else:

                d_input[i][j] = 0

    return d_input


# ==================================================
# Conv Backward
# ==================================================

def conv2d_backward(
    d_output,
    input_image,
    kernel
):

    h, w = input_image.shape

    kh, kw = kernel.shape

    d_input = np.zeros(
        (h, w)
    )

    d_kernel = np.zeros(
        (kh, kw)
    )

    output_height, output_width = d_output.shape

    for i in range(output_height):

        for j in range(output_width):

            gradient = d_output[i][j]

            for x in range(kh):

                for y in range(kw):

                    d_kernel[x][y] += (
                        input_image[i + x][j + y]
                        * gradient
                    )

                    d_input[i + x][j + y] += (
                        kernel[x][y]
                        * gradient
                    )

    return d_input, d_kernel


# ==================================================
# MNIST
# ==================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train[:500]
y_train = y_train[:500]

x_test = x_test[:1000]
y_test = y_test[:1000]


x_train = x_train / 255.0
x_test = x_test / 255.0


# ==================================================
# 参数初始化
# ==================================================

kernel = np.random.randn(3, 3) * 0.01

# 13×13 = 169
W = np.random.randn(169, 10) * 0.01

b = np.zeros(10)


learning_rate = 0.01


# ==================================================
# Training
# ==================================================

for epoch in range(5):

    total_loss = 0


    for i in range(500):

        # ==========================================
        # 取一张图片
        # ==========================================

        image = x_train[i]

        label = y_train[i]


        # ==========================================
        # Forward
        # ==========================================

        feature_map = conv2d(
            image,
            kernel
        )

        relu_output = relu(
            feature_map
        )

        pooled_map, max_positions = max_pool(
            relu_output
        )

        flattened_map = flatten(
            pooled_map
        )

        output = np.dot(
            flattened_map,
            W
        ) + b


        # ==========================================
        # Target
        # ==========================================

        target = np.zeros(10)

        target[label] = 1


        # ==========================================
        # Loss
        # ==========================================

        loss = np.mean(
            (output - target) ** 2
        )

        total_loss += loss


        # ==========================================
        # Backward
        # ==========================================

        d_output = 2 * (
            output - target
        ) / 10


        # Linear Backward

        d_flat = np.dot(
            d_output,
            W.T
        )

        d_W = np.outer(
            flattened_map,
            d_output
        )

        d_b = d_output


        # Flatten Backward

        d_pool = flatten_backward(
            d_flat,
            pooled_map.shape
        )


        # Max Pool Backward

        d_relu = max_pool_backward(
            d_pool,
            max_positions,
            relu_output.shape
        )


        # ReLU Backward

        d_conv = relu_backward(
            d_relu,
            feature_map
        )


        # Conv Backward

        d_image, d_kernel = conv2d_backward(
            d_conv,
            image,
            kernel
        )


        # ==========================================
        # Update
        # ==========================================

        W = W - learning_rate * d_W

        b = b - learning_rate * d_b

        kernel = kernel - learning_rate * d_kernel


    print(
        "epoch:",
        epoch,
        "loss:",
        total_loss / 5000
    )