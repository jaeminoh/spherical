import numpy as np
from sklearn.linear_model import LassoCV


def _normalization(x_train, y_train):
    x_mean, x_std = (
        np.mean(x_train, axis=0, keepdims=True),
        np.std(x_train, axis=0, keepdims=True),
    )
    y_mean, y_std = np.mean(y_train), np.std(y_train)

    def normalize(x, y):
        return (x - x_mean) / (x_std + 1e-3), (y - y_mean) / (y_std + 1e-3)

    return normalize, (x_mean, x_std), (y_mean, y_std)


data = np.load("data.npz")
x = data["ag"] * data["hchain"] * data["lchain"]
y = data["dG"]

x_train = x[:150]
y_train = y[:150]

normalize, *_ = _normalization(x_train, y_train)

x_test = x[150:]
y_test = y[150:]

reg = LassoCV(cv=5).fit(*normalize(x_train, y_train))
print(f"R^2 for train data: {reg.score(*normalize(x_train, y_train))}")
print(f"R^2 for test data: {reg.score(*normalize(x_test, y_test))}")
