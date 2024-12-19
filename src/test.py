# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ali = "ali"

# %%
data = pd.read_csv(
    "C:\\Users\\7862s\\Desktop\\AI PROJECT\\datasets\\train.csv\\train.csv"
)

# %%
data.head()

# %%
array_data = np.array(data)
m, n = array_data.shape
np.random.shuffle(array_data)

data_dev = array_data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = array_data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


# %%
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T


def back_prop(z1, a1, z2, a2, w2, x, y):
    m = y.size
    one_hot_y = one_hot(y)
    delta_z2 = a2 - one_hot_y
    delta_w2 = 1 / m * delta_z2.dot(a1.T)
    delta_b2 = 1 / m * np.sum(delta_z2)
    delta_z1 = w2.dot(delta_z2) * (z1 > 0)  # derivative for linear function
    delta_w1 = 1 / m * delta_z1.dot(x.T)
    delta_b1 = 1 / m * np.sum(delta_z1)
    return delta_w1, delta_b1, delta_w2, delta_b2


def update_params(w1, b1, w2, b2, del_w1, del_b1, del_w2, del_b2, alpha):
    w1 -= alpha * del_w1
    b1 -= alpha * del_b1
    w2 -= alpha * del_w2
    b2 -= alpha * del_b2
    return w1, b1, w2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        delta_z1, delta_b1, delta_z2, delta_b2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(
            w1, b1, w2, b2, delta_z1, delta_b1, delta_z2, delta_b2, alpha
        )
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(predictions, y)
            print(get_accuracy(predictions, y))

    return w1, b1, w2, b2


# %%
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
