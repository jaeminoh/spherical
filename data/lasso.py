import numpy as np
from sklearn.linear_model import LassoCV

data = np.load("data.npz")
x = data["ag"] * data["hchain"] * data["lchain"]
y = data["dG"]

x_train = x[:150]
y_train = y[:150]
x_test = x[150:]
y_test = y[150:]

reg = LassoCV(cv=5).fit(x_train, y_train)
print(f"R^2 for train data: {reg.score(x_train, y_train)}")
print(f"R^2 for test data: {reg.score(x_test, y_test)}")
