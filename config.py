from dataclasses import dataclass
from typing import Any


@dataclass
class Traindata:
    name: str
    path: str
    target_path: str


@dataclass
class Testdata:
    name: str
    path: str
    target_path: str


@dataclass
class Model:
    k_neighbours: int
    weight_samples: bool
    weight_param: float


@dataclass
class Training:
    file_path: str


@dataclass
class Params:
    data: Any
    model: Model
    training: Training
