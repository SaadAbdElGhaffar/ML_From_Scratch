import numpy as np


def mini_batch_gradient_descent(alpha, x, y, batch_size=20, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    while not converged:
        for i in range(0, m, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            grad0 = (
                1.0
                / len(y_batch)
                * sum(
                    [(t0 + t1 * x_batch[j] - y_batch[j]) for j in range(len(y_batch))]
                )
            )
            grad1 = (
                1.0
                / len(y_batch)
                * sum(
                    [
                        (t0 + t1 * x_batch[j] - y_batch[j]) * x_batch[j]
                        for j in range(len(y_batch))
                    ]
                )
            )

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
            converged = True

    return t0, t1
