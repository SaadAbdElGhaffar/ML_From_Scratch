import numpy as np


def stochastic_gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    while not converged and iter < max_iter:
        for i in range(m):
            grad0 = t0 + t1 * x[i] - y[i]
            grad1 = grad0 * x[i]

            t0 -= alpha * grad0
            t1 -= alpha * grad1

        e = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print("Converged, iterations:", iter, "!!!")
            converged = True
        J = e
        iter += 1

    if iter == max_iter:
        print("Max iterations exceeded!")

    return t0, t1
