from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, classification_report

def calc_mse_mae(preds, y_test):
    MSE=mean_squared_error(preds, y_test)
    MAE=mean_absolute_error(preds, y_test)
    print('Mean squared Error: ', MSE, ' Mean Absolute Error: ', MAE)

def get_accuracy(ideal_gt, segmented_img):
  n_pixels = ideal_gt.shape[0] * ideal_gt.shape[1]
  n_correct = 0
  for i in range(ideal_gt.shape[0]):
    for j in range(ideal_gt.shape[1]):
        if (ideal_gt[i, j, :] == segmented_img[i, j, :]).all():
            n_correct += 1
  accuracy = n_correct / n_pixels * 100
  return accuracy


def f1_metric(true, predictions, average=None):
   f1 = f1_score(true, predictions, average=average)
   return f1

def get_classification_report(true, predictions, digits=3):
   class_report = classification_report(true, predictions, digits=digits)
   return class_report

