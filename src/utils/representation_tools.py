import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from src.utils.metrics_tools import calc_mse_mae

FEATURES = ['R','G', 'B', 'NIR1', 'NIR2', 'NIR3','EVI1','EVI3','EVI3']

class RepresentationTools:


    def plot_linear_regression_pca(matrix_pca):
        '''
        Input matrix with pca already applied
        '''
        X = matrix_pca[:-1, :]
        Y = matrix_pca[-1, :]
        model= linear_model.LinearRegression()

        for i, feat in enumerate(FEATURES):
            if len(FEATURES) == X.shape[0]:
                X_train, X_test, y_train, y_test = train_test_split(np.transpose(X[i,:]), np.squeeze(Y), test_size=0.2, random_state=1234)
                X_train = X_train.reshape(-1,1)
                y_train = y_train.reshape(-1,1)
                trained_model=model.fit(X_train, y_train)
                X_test = X_test.reshape(-1,1)
                predictions = trained_model.predict(X_test)

                calc_mse_mae(predictions, y_test)
                print(f"Coefficients: {trained_model.coef_}")

                plt.figure(figsize=(10, 10))
                plt.scatter(X_test, np.squeeze(y_test), color="black")
                plt.plot(X_test, predictions, color="blue", linewidth=2)
                plt.title(f"{feat} vs GT")
                plt.show()