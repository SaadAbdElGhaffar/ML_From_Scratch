import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import math
from numpy import linalg as LA
cvxopt_solvers.options['show_progress'] = False

class SVRCustom:
    def __init__(self, C=5, gamma=0.01, eps=0.01):
        self.x_train = None
        self.y_train = None
        self.C = C
        self.gamma = gamma
        self.eps = eps
        self.b = 0
        self.alpha = None
        self.alpha_star = None

    def kernel_rbf(self, x, y, gamma):
        return math.exp((-1) * gamma * (LA.norm(x - y) ** 2))

    def kernel_matrix_rbf(self, X, gamma):
        n = len(X)
        k = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                k[i][j] = self.kernel_rbf(X[i], X[j], gamma)
        return k

    def P_Matrix_rbf(self, X, gamma):
        k = self.kernel_matrix_rbf(X, gamma)
        neg_k = (-1) * k
        temp1 = np.concatenate((k, neg_k))
        temp2 = np.concatenate((neg_k, k))
        P = np.concatenate((temp1, temp2), axis=1)
        return P

    def q_matrix(self, y, eps):
        temp1 = -y + eps
        temp2 = y + eps
        q = np.concatenate((temp1, temp2))
        return q

    def G_matrix(self, n):
        temp1 = np.identity(2 * n)
        temp2 = (-1.0) * temp1
        G = np.concatenate((temp1, temp2))
        return G

    def h_matrix(self, C, n):
        temp1 = C * np.ones(2 * n)
        temp2 = np.zeros(2 * n)
        h = np.concatenate((temp1, temp2))
        return h

    def A_matrix(self, n):
        temp1 = np.ones(n)
        temp2 = (-1.0) * temp1
        A = np.concatenate((temp1, temp2))
        A = A.reshape(1, -1)
        return A

    def b_matrix(self):
        return np.zeros(1)

    def b_term_rbf(self, x_support_vector, n, alpha, alpha_star, x_train, gamma):
        total = 0
        for i in range(n):
            total += (alpha[i][0] - alpha_star[i][0]) * self.kernel_rbf(x_support_vector, x_train[i], gamma)
        return total

    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train

        n = len(X_train)
        P = self.P_Matrix_rbf(X_train, self.gamma)
        q = self.q_matrix(y_train, self.eps)
        G = self.G_matrix(n)
        h = self.h_matrix(self.C, n)
        A = self.A_matrix(n)
        b = self.b_matrix()

        P = cvxopt_matrix(P)
        q = cvxopt_matrix(q)
        G = cvxopt_matrix(G)
        h = cvxopt_matrix(h)
        A = cvxopt_matrix(A)
        b = cvxopt_matrix(b)

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        l = int(len(alphas) / 2)
        self.alpha = alphas[0:l, :]
        self.alpha_star = alphas[l:, :]

        supp_vector = []
        for j in range(len(self.alpha)):
            current_alpha_value = self.alpha[j][0]
            if 1e-5 < current_alpha_value < self.C:
                supp_vector.append(j)

        Y = y_train.flatten()
        self.b = 0
        for i in supp_vector:
            self.b += Y[i] - self.eps - self.b_term_rbf(X_train[i], n, self.alpha, self.alpha_star, X_train, self.gamma)
        self.b /= len(supp_vector)

    def Y_predicted_rbf(self, X_to_be_predicted, n, alpha, alpha_star, X_train, b, gamma):
        total = 0
        for i in range(n):
            total += (alpha[i][0] - alpha_star[i][0]) * self.kernel_rbf(X_to_be_predicted, X_train[i], gamma)
        y = total + b
        return y

    def predicted(self, X_to_be_predicted):
        n = len(self.x_train)
        predictions_SVR = []
        for i in range(len(self.x_train)):
            current_point_pred = self.Y_predicted_rbf(self.x_train[i], n, self.alpha, self.alpha_star, self.x_train, self.b, self.gamma)
            predictions_SVR.append(current_point_pred)
        predictions_SVR = np.array(predictions_SVR)
        return predictions_SVR
