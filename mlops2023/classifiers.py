from collections import Counter

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore

from config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    pass


class KNNClassifier(object):
    def __init__(
        self, k_neighbours: int, weight_samples: bool, weight_param: int
    ):  # noqa

        self._k_neighbours = k_neighbours
        self._weight_samples = weight_samples
        self._weight_param = weight_param

        self._X, self._y = None, None

    def fit(self, X: np.array, y: np.array) -> None:
        '''
        When fit() method called -- model just saves the Xs and ys
        '''
        self._X = X
        self._y = y

    def predict(self, X: np.array) -> np.array:
        '''Non-optimized version (python loop-based)'''

        # Assertion check -- if model is fitted or not
        assert (
            self._X is not None and self._y is not None
        ), f"Model is not fitted yet!"  # noqa

        ys_pred: np.array = np.zeros(
            shape=(X.shape[0], 1)
        )  # Predictions matrix allocation

        '''
        For each sample in X calculate distances to the points in self._X, using the self._metric() # noqa
        calculate distances and get K nearest points.
        '''
        for sample_id, X_this in enumerate(X):
            distances = dict(
                enumerate(np.sqrt(np.sum((self._X - X_this) ** 2, axis=1)))
            )  # noqa
            sorted_distances = self._sort_dict(distances)
            y_pred: int = self._get_nearest_class(sorted_distances)
            ys_pred[sample_id] = y_pred

        return ys_pred

    @staticmethod
    def _sort_dict(unsort_dict, ascending=False):
        return dict(sorted(unsort_dict.items(), key=lambda item: item[1]))

    def _get_nearest_class(self, sorted_distances: list) -> int:
        sorted_distances_top_k = list(sorted_distances.keys())[
            : self._k_neighbours
        ]  # noqa
        labels_top_k = [
            dict(enumerate(self._y))[sample] for sample in sorted_distances_top_k  # noqa
        ]
        predicted_label: int = self._decision_rule(labels_top_k)
        return predicted_label

    def _decision_rule(self, labels_top_k) -> int:
        if self._weight_samples:

            # Создаем словарь для подсчета весов по классам
            class_weights = {}
            weights = np.power(
                self._weight_param, np.arange(len(labels_top_k))
            )  # пусть вес убывает экспоненциально
            # Суммируем веса для каждого класса
            for label, weight in zip(labels_top_k, weights):
                class_weights[label] = class_weights.get(label, 0) + weight

            # Находим класс с наибольшей суммой весов
            max_weight_class = max(class_weights, key=class_weights.get)
            return max_weight_class

        else:
            return Counter(labels_top_k).most_common()[0][0]


if __name__ == "__main__":
    main()
