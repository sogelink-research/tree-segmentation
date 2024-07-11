from __future__ import annotations

import json
import os
import pickle
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rich.traceback import install

from model_session import (
    DatasetParams,
    FullJsonEncoder,
    ModelSession,
    TrainingData,
    TrainingParams,
)
from utils import RICH_PRINTING, Folders, create_all_folders


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
        experiment: str = "exp0",
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
        self.experiment = experiment
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
        self.init_postfix = postfix

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

    def _init_training_params(self) -> None:

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
            experiment=self.experiment,
        )
        training_data = TrainingData(
            dataset_params=self.dataset_params, training_params=training_params
        )
        postfix = (
            self.init_postfix + self.experiment
            if self.init_postfix is not None
            else self.experiment
        )
        super().__init__(training_data=training_data, device=self.device, postfix=postfix)

        self.save_init_params()

    def train(self) -> None:
        self._init_training_params()
        return super().train()

    # def compute_metrics(self, initialize: bool = True):
    #     self._init_training_params()
    #     return super().compute_metrics(initialize)

    @property
    def init_params_path(self) -> str:
        return os.path.join(self.folder_path, "model_init_params.json")

    def save_init_params(self) -> None:
        params_to_save = {
            "use_rgb": self.use_rgb,
            "use_cir": self.use_cir,
            "use_chm": self.use_chm,
            "chm_z_layers": self.chm_z_layers,
            "annotations_file_name": self.annotations_file_name,
            "agnostic": self.agnostic,
            "experiment": self.experiment,
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
            model_session: ModelTrainingSession = pickle.load(f)
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


class ParamsCombinations:

    def __init__(
        self,
        name: str,
        params_dict: Optional[Dict[str, List]] = None,
        forget_combinations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:

        self.name = name

        if os.path.exists(self.state_path):
            self.load_state()
            return

        if params_dict is None or forget_combinations is None:
            raise Exception(
                "If a ParamsCombinations object with the same name doesn't exist, every parameter should be specified."
            )

        keys, values = zip(*params_dict.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        self.combinations = [
            comb
            for comb in combinations
            if not any(
                [
                    all([comb[key] == value for key, value in forget_comb.items()])
                    for forget_comb in forget_combinations
                ]
            )
        ]

        self.model_names = [""]
        self.next_idx = 0
        self.save_state()

    def save_state(self) -> None:
        state = {"next_idx": self.next_idx, "parameters": []}
        for idx, combination in enumerate(self.combinations):
            if idx < len(self.model_names):
                model_name = self.model_names[idx]
            else:
                model_name = ""
            state["parameters"].append(
                {
                    "combination": combination,
                    "model_name": model_name,
                }
            )
        with open(self.state_path, "w") as fp:
            json.dump(state, fp, cls=FullJsonEncoder, sort_keys=True)

    def load_state(self) -> None:
        with open(self.state_path, "r") as file:
            state = json.load(file)

        self.next_idx = state["next_idx"]
        parameters = state["parameters"]
        self.combinations = list(map(lambda p: p["combination"], parameters))
        self.model_names = list(map(lambda p: p["model_name"], parameters))

    @property
    def state_path(self) -> str:
        state_path = os.path.join(Folders.MODELS_EXPERIMENTS.value, f"{self.name}.json")
        return state_path

    def __iter__(self):
        total_combinations = len(self.combinations)
        for idx in range(self.next_idx, len(self.combinations)):
            str_len = len(str(total_combinations))
            RICH_PRINTING.print(f"Experiment {idx+1:>{str_len}}/{total_combinations}")

            model_training_session = ModelTrainingSession(**self.combinations[idx])
            yield model_training_session

            self.model_names[idx] = model_training_session.model_name
            self.next_idx += 1
            self.save_state()


def main():
    # install(show_locals=True)
    create_all_folders()

    params_dict = {
        "epochs": [1],
        "experiment": ["exp0"],
        "lr": [1e-2, 6e-3, 2.5e-3, 1e-3],
        "accumulate": [6, 12, 18, 24, 36],
        "proba_drop_rgb": [0, 0.1, 0.2, 0.333],
        "model_size": ["n"],
        "agnostic": [True],
        "use_rgb": [True],
        "use_cir": [True],
        "use_chm": [True],
    }

    forget_combinations = [
        {
            "use_rgb": False,
            "use_cir": False,
            "use_chm": False,
        },
    ]

    params_combinations = ParamsCombinations(
        "training_params_experiment",
        params_dict=params_dict,
        forget_combinations=forget_combinations,
    )

    # for model_training_session in params_combinations:
    #     # Training session
    #     model_training_session.train()

    names = os.listdir("models/amf_gd_yolov8")
    for name in RICH_PRINTING.pbar(names, len(names), description="Trained models", leave=True):
        model_training_session = ModelTrainingSession.from_name(name)
        # model_training_session.model_name = name
        model_training_session.compute_metrics()


if __name__ == "__main__":
    main()
    RICH_PRINTING.close()
