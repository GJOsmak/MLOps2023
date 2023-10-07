import joblib
import pandas as pd
import numpy as np
from mlops2023 import classifiers
file_path = './trained_model.joblib'

X = np.array(pd.read_csv('./data/mnist_train.csv').iloc[:,1:])
y = np.array(pd.read_csv('./data/mnist_target_train.csv').iloc[:,1])

clf = classifiers.KNNClassifier(10, weight_samples=False)
clf.fit(X, y)
joblib.dump(clf, file_path)
print("Модель успешно обучена и сохранена в", file_path)
