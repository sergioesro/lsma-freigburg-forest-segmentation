from sklearn.metrics import mean_squared_error, mean_absolute_error

def calc_mse_mae(preds, y_test):
    MSE=mean_squared_error(preds, y_test)
    MAE=mean_absolute_error(preds, y_test)
    print('Mean squared Error: ', MSE, ' Mean Absolute Error: ', MAE)