import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt


def model(x, u):
    return np.sum(x*u, 1)


def fun(x, u, y):
    return model(x, u) - y


def jac(x, u, y):
    return u


dataName = "Peak_All"

A = np.loadtxt("/mnt/d/003_Research/20200900_Bkg/" + dataName + "_Data.txt")
B = np.loadtxt("/mnt/d/003_Research/20200900_Bkg/" + dataName + "_Meas.txt")
E = np.loadtxt("/mnt/d/003_Research/20200900_Bkg/Energies.txt")

A = np.transpose(A)
N = A.shape[1]


def res(x):
    return np.sum((np.sum(x*A, 1) - B) ** 2)


x0 = np.ones(N)

# print(A.shape)
# print(np.sum(np.ones(A.shape[1])*A, 1))
# print(model(np.ones(A.shape[1]), A))
# print(fun(np.ones(A.shape[1]), A, B))
# print(jac(np.ones(A.shape[1]), A, B))

bounds = Bounds(np.zeros(N), np.inf*np.ones(N))
linBoundsArray = np.eye(N-1, N) - np.diag(np.ones(N-1), k=1)[:-1, :]
# linBoundsArray = -np.eye(N-1, N) + np.diag(np.ones(N-1), k=1)[:-1, :]
linear_constraint = LinearConstraint(linBoundsArray, -np.inf*np.ones(N-1), np.zeros(N-1))

res = minimize(res, x0, method='trust-constr', bounds=bounds, constraints=[linear_constraint], options={'verbose': 1})
C = res.x

# C = np.linalg.lstsq(A, B, rcond=None)[0]
print(C)
np.savetxt("/mnt/d/003_Research/20200900_Bkg/" + dataName + "_rslt_inc.txt", C)

plt.scatter(E, C)
# plt.scatter(np.arange(np.size(A, 0)), B)
# plt.plot(np.dot(A, C))
plt.show()
