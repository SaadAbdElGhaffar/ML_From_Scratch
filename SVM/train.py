import pandas as pd
from SVR import SVRCustom
from sklearn.svm import SVR

train_data = pd.read_csv('train_data.csv')

X = train_data.drop(columns=['SalePrice'])
y = train_data[['SalePrice']]
c = 100
gamma = 0.1
eps = 0.1

modelCustom = SVRCustom(c, gamma, eps)
modelCustom.fit(X.values, y.values.flatten())
predCustom = modelCustom.predicted(X.values)
print(predCustom)

svr_sklearn = SVR(kernel='rbf', C=c, gamma=gamma, epsilon=eps)

fitted_SVR_Sklearn = svr_sklearn.fit(X.values, y.values.flatten())
pred = fitted_SVR_Sklearn.predict(X.values)
print(pred)
