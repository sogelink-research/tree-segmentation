from __future__ import annotations

import json
import os
import pickle
import warnings
from itertools import product
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch

from model_session import (
    DatasetParams,
    FullJsonEncoder,
    ModelSession,
    TrainingData,
    TrainingParams,
)
from utils import RICH_PRINTING


class ModelTrainingSession(ModelSession):
    """A class to launch a training session with only the necessary hyperparameters"""

    def __init__(
        self,
        use_rgb: bool = True,
        use_cir: bool = True,
        use_chm: bool = True,
        chm_z_layers: Optional[Sequence[Tuple[float, float]]] = None,
        annotations_file_name: str = "122000_484000.geojson",
        agnostic: bool = False,
        experiences: List[str] = ["exp0", "exp1", "exp2", "exp3", "exp4"],
        model_size: str = "n",
        lr: float = 1e-2,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        accumulate: int = 12,
        no_improvement_stop_epochs: int = 50,
        proba_drop_rgb: float = 1 / 3,
        proba_drop_chm: float = 1 / 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        postfix: Optional[str] = None,
    ) -> None:
        self.use_rgb = use_rgb
        self.use_cir = use_cir
        self.use_chm = use_chm
        self.chm_z_layers = chm_z_layers
        self.annotations_file_name = annotations_file_name
        self.agnostic = agnostic
        self.experiences = experiences
        self.model_size = model_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulate = accumulate
        self.no_improvement_stop_epochs = no_improvement_stop_epochs
        self.proba_drop_rgb = proba_drop_rgb
        self.proba_drop_chm = proba_drop_chm
        self.device = device
        self.postfix = postfix

        self._init_dataset_params()

    def _init_dataset_params(self) -> None:

        if self.chm_z_layers is None:
            z_tops = [1, 2, 3, 5, 7, 10, 15, 20, np.inf]
            chm_z_layers = [(-np.inf, z_top) for z_top in z_tops]

        self.dataset_params = DatasetParams(
            annotations_file_name=self.annotations_file_name,
            use_rgb=self.use_rgb,
            use_cir=self.use_cir,
            use_chm=self.use_chm,
            chm_z_layers=chm_z_layers,
            agnostic=self.agnostic,
        )

    def _init_training_params(self, experience: str) -> None:

        training_params = TrainingParams(
            model_size=self.model_size,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            accumulate=self.accumulate,
            no_improvement_stop_epochs=self.no_improvement_stop_epochs,
            proba_drop_rgb=self.proba_drop_rgb,
            proba_drop_chm=self.proba_drop_chm,
            experience=experience,
        )
        training_data = TrainingData(
            dataset_params=self.dataset_params, training_params=training_params
        )
        postfix = self.postfix + experience if self.postfix is not None else experience
        super().__init__(training_data=training_data, device=self.device, postfix=postfix)

        self.save_init_params(experience)

    def train(self) -> None:
        for experience in self.experiences:
            self._init_training_params(experience)
            super().train()

    @property
    def init_params_path(self) -> str:
        return os.path.join(self.folder_path, "model_init_params.json")

    def save_init_params(self, experience: str) -> None:
        params_to_save = {
            "use_rgb": self.use_rgb,
            "use_cir": self.use_cir,
            "use_chm": self.use_chm,
            "chm_z_layers": self.chm_z_layers,
            "annotations_file_name": self.annotations_file_name,
            "agnostic": self.agnostic,
            "experience": experience,
            "model_size": self.model_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "accumulate": self.accumulate,
            "no_improvement_stop_epochs": self.no_improvement_stop_epochs,
            "proba_drop_rgb": self.proba_drop_rgb,
            "proba_drop_chm": self.proba_drop_chm,
            "device": self.device,
            "postfix": self.postfix,
        }
        save_path = self.init_params_path
        with open(save_path, "w") as fp:
            json.dump(params_to_save, fp, cls=FullJsonEncoder, sort_keys=True, indent=4)

    @staticmethod
    def from_pickle(file_path: str, device: torch.device) -> ModelTrainingSession:
        with open(file_path, "rb") as f:
            model_session = pickle.load(f)
            model_session.device = device
        return model_session

    @staticmethod
    def from_name(
        model_name: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> ModelTrainingSession:
        if not ModelTrainingSession.already_exists(model_name):
            raise Exception(f"There is no model called {model_name}.")
        file_path = ModelTrainingSession.get_pickle_path(model_name)
        return ModelTrainingSession.from_pickle(file_path, device)


def main():
    params_dict = {
        "agnostic": [True, False],
        "use_rgb": [True, False],
        "use_cir": [True, False],
        "use_chm": [True, False],
        "lr": [1e-2, 3e-3, 1e-3],
        "model_size": ["n"],
        "epochs": [1],
    }

    forget_combinations = [
        {
            "use_rgb": False,
            "use_cir": False,
            "use_chm": False,
        },
    ]

    # Generate all combinations of arguments
    keys, values = zip(*params_dict.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    filtered_combinations = [
        comb
        for comb in combinations
        if not any(
            [
                all([comb[key] == value for key, value in forget_comb.items()])
                for forget_comb in forget_combinations
            ]
        )
    ]

    for combination in filtered_combinations:
        # Training session
        model_training_session = ModelTrainingSession(**combination)
        model_training_session.train()

    # model_training_session = ModelTrainingSession(epochs=0)
    # model_training_session.train()

    # for name in os.listdir("models/amf_gd_yolov8"):
    #     model_training_session = ModelTrainingSession.from_name(name)
    #     model_training_session.compute_metrics()


if __name__ == "__main__":
    main()
    RICH_PRINTING.close()
