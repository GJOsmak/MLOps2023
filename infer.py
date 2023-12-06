import hydra
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):

    X_test = np.array(pd.read_csv(cfg.data.test_data).iloc[:, 1:])
    y_test = np.array(pd.read_csv(cfg.data.test_target).iloc[:, 1])

    knn_model = joblib.load(cfg.paths.trained_model)
    y_pred = knn_model.predict(X_test)
    np.savetxt(cfg.paths.result, y_pred, delimiter=',', fmt='%d')
    print('Classifier Accuracy =', accuracy_score(y_test, y_pred))
    print('Предсказания сохранены в', cfg.paths.result)
