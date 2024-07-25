import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# calculating the economy svd (where the u matrix has the same size as the data matrix X, nad the sigma matrix is square matrix)
# AKA: no silent vectors in the u Matrix, so it is not square and not orthogonal (uTu = I, but uuT is != I)
def linear_reg_SVD(X, Y, zero_threshold = 1e-13):
  u, s, vT = np.linalg.svd(X, full_matrices= False)

  # Now initialize the "pseudo-"inverse of Sigma, where "pseudo" means "don't divide by zero"
  sigma_pseudo_inverse = np.zeros((vT.shape[0], vT.shape[0]))

  ## getting the index of the first approximately zero singular value
  idx_nearly_zero_sigma= np.where(s <= zero_threshold)[0]
  if len(idx_nearly_zero_sigma) > 0:
    # 1/non-zero diagonal elements calculation
    sigma_pseudo_inverse[:idx_nearly_zero_sigma[0],:idx_nearly_zero_sigma[0]] = np.diag(1/s[ :idx_nearly_zero_sigma[0]])
  else:
      sigma_pseudo_inverse[:len(s),:len(s)] = np.diag(1/s)
  #the above three lines could have been calculated via:
  # sigma_pseudo_inverse = = np.linalg.pinv(np.diag(s), rcond=1e-13)

  # calculating the optimal coefficients
  optimal_coefficients = vT.T.dot(sigma_pseudo_inverse).dot(u.T).dot(Y)
  return optimal_coefficients

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame for easier manipulation and exploration
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Separate the features (X) and the target variable (y)
X = iris_df.drop(columns = ['target'])
y = iris_df.target


# Split the dataset into training and testing sets
# By default, train_test_split uses 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

optimal_coefficients = linear_reg_SVD(X_train, y_train, zero_threshold = 1e-13)
y_hat_from_scratch = X_train.values.dot(optimal_coefficients)

# linear regression from sklearn
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

y_hat_sklearn = reg.predict(X_train)

# Compare predictions using mean squared error
mse_custom = mean_squared_error(y_train, y_hat_from_scratch)
mse_sklearn = mean_squared_error(y_train, y_hat_sklearn)

print(f"Custom MSE: {mse_custom:.4f}")
print(f"Scikit-Learn MSE: {mse_sklearn:.4f}")