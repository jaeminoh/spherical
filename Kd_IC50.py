from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

Kd = np.array([153, 47.7, 295, 18.5, 323, 743, 32.8, 20.5])
IC50 = np.array([46.8, 1.85, 56.2, 0.53, 0.29, 2.53, 51.1, 73.7])

x = Kd.reshape(-1, 1)
y = IC50

reg = LinearRegression().fit(x, y)
print(f"R^2 for train data: {reg.score(x, y)}")
print(f"Intercept: {reg.intercept_}")
print(f"Coefficient: {reg.coef_}")
plt.figure(figsize=(4,3))
plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, reg.predict(x), color="red", label="Regression Line")
plt.xlabel("Kd")
plt.ylabel("IC50")
plt.title("Kd vs IC50, " + r"$R^2$ = " + f"{reg.score(x, y):.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("Kd_vs_IC50.pdf")


