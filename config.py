from dataclasses import dataclass


@dataclass
class Data:
    test_data: str
    test_target: str
    train_data: str
    train_target: str


@dataclass
class Model:
    k_neighbours: int
    weight_samples: bool
    weight_param: float


@dataclass
class Paths:
    trained_model: str
    result: str


@dataclass
class KNNMnist_param:
    data: Data
    model: Model
    paths: Paths
