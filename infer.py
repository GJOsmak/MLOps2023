import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

file_path = './trained_model.joblib'
relut_file = './prediction.csv'

X_test = np.array(pd.read_csv('./data/mnist_test.csv').iloc[:,1:])
y_test = np.array(pd.read_csv('./data/mnist_target_test.csv').iloc[:,1])

knn_model = joblib.load(file_path)
y_pred = knn_model.predict(X_test)
np.savetxt(relut_file, y_pred, delimiter=',', fmt='%d')
print('Classifier Accuracy =', accuracy_score(y_test, y_pred))
print('Предсказания сохранены в', relut_file)
