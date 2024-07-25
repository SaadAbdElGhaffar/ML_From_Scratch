import numpy as np
from sklearn.datasets.samples_generator import make_regression

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    J = sum([(t0 + t1 * x[i] - y[i])**2 for i in range(m)])

    while not converged:
        grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        t0 = temp0
        t1 = temp1

        e = sum([(t0 + t1 * x[i] - y[i])**2 for i in range(m)])

        if abs(J - e) <= ep:
            print('Converged, iterations:', iter, '!!!')
            converged = True
        J = e
        iter += 1

        if iter == max_iter:
            print('Max iterations exceeded!')
            converged = True

    return t0, t1

if __name__ == '__main__':
    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)
    alpha = 0.01  # learning rate
    ep = 0.01  # convergence criteria
    theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1000)
    print('Intercept (theta0):', theta0)
    print('Slope (theta1):', theta1)
