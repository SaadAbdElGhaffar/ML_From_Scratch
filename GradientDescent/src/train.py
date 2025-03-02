import numpy as np
from sklearn.datasets import make_regression
from src.batch_gradient_descent import batch_gradient_descent
from src.stochastic_gradient_descent import stochastic_gradient_descent
from src.mini_batch_gradient_descent import mini_batch_gradient_descent


def main():
    # Generate synthetic data
    x, y = make_regression(
        n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35
    )

    # Parameters
    alpha = 0.01  # learning rate
    ep = 0.01  # convergence criteria
    max_iter = 1000

    # Train using standard gradient descent
    theta0, theta1 = batch_gradient_descent(alpha, x, y, ep, max_iter)
    print("Standard Gradient Descent:")
    print("Intercept (theta0):", theta0)
    print("Slope (theta1):", theta1)

    # Train using stochastic gradient descent
    theta0_sgd, theta1_sgd = stochastic_gradient_descent(alpha, x, y, ep, max_iter)
    print("Stochastic Gradient Descent:")
    print("Intercept (theta0):", theta0_sgd)
    print("Slope (theta1):", theta1_sgd)

    # Train using mini-batch gradient descent
    batch_size = 10
    theta0_mbgd, theta1_mbgd = mini_batch_gradient_descent(
        alpha, x, y, ep, max_iter, batch_size
    )
    print("Mini-Batch Gradient Descent:")
    print("Intercept (theta0):", theta0_mbgd)
    print("Slope (theta1):", theta1_mbgd)


if __name__ == "__main__":
    main()
