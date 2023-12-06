import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore

from config import KNNMnist_param
from mlops2023 import classifiers


cs = ConfigStore.instance()
cs.store(name="knn_mnist_param", node=KNNMnist_param)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):

    X = np.array(pd.read_csv(cfg.data.train_data).iloc[:, 1:])
    y = np.array(pd.read_csv(cfg.data.train_target).iloc[:, 1])

    clf = classifiers.KNNClassifier(**cfg.model)
    clf.fit(X, y)
    joblib.dump(clf, cfg.paths.trained_model)
    print("Модель успешно обучена и сохранена в", cfg.paths.trained_model)


if __name__ == "__main__":
    main()
