from __future__ import annotations

import json
import os
import pickle
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from layers import AMF_GD_YOLOv8
from model_session import (
    DatasetParams,
    FullJsonEncoder,
    ModelSession,
    TrainingData,
    TrainingParams,
)
from utils import RICH_PRINTING, Folders, create_all_folders, create_folder


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
        repartition_name: str = "exp0",
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
        prefix: Optional[str] = None,
        experiment_name: str = "",
    ) -> None:
        self.use_rgb = use_rgb
        self.use_cir = use_cir
        self.use_chm = use_chm
        self.chm_z_layers = chm_z_layers
        self.annotations_file_name = annotations_file_name
        self.agnostic = agnostic
        self.repartition_name = repartition_name
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
        self.init_prefix = prefix
        self.experiment_name = experiment_name

        if experiment_name != "":
            self.parent_folder_path = os.path.join(
                Folders.MODELS_EXPERIMENTS.value, experiment_name
            )
        else:
            self.parent_folder_path = Folders.MODELS_AMF_GD_YOLOV8.value

        create_folder(self.parent_folder_path)

        self._init_dataset_params()
        self._init_training_params()

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
            repartition_name=self.repartition_name,
        )
        training_data = TrainingData(
            dataset_params=self.dataset_params, training_params=training_params
        )
        prefix = (
            self.init_prefix + self.repartition_name
            if self.init_prefix is not None
            else self.repartition_name
        )
        super().__init__(
            training_data=training_data,
            device=self.device,
            prefix=prefix,
            parent_folder_path=self.parent_folder_path,
        )

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
            "repartition_name": self.repartition_name,
            "model_size": self.model_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": None,  ### TO MODIFY
            "num_workers": self.num_workers,
            "accumulate": self.accumulate,
            "no_improvement_stop_epochs": self.no_improvement_stop_epochs,
            "proba_drop_rgb": self.proba_drop_rgb,
            "proba_drop_chm": self.proba_drop_chm,
            "device": self.device,
            "prefix": self.prefix,
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
        parent_folder_path: str,
        model_name: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> ModelTrainingSession:
        if not ModelTrainingSession.already_exists(
            parent_folder_path=parent_folder_path, model_name=model_name
        ):
            raise Exception(f"There is no model called {model_name}.")
        file_path = ModelTrainingSession.get_pickle_path(
            parent_folder_path=parent_folder_path, model_name=model_name
        )
        return ModelTrainingSession.from_pickle(file_path, device)

    # def update(self, experiment_name: str) -> None:
    #     old_model_name = self.model_name
    #     self.postfix = self.postfix if self.postfix is not None else ""
    #     self.prefix = self.postfix
    #     self.model_name = AMF_GD_YOLOv8.get_new_name(prefix=self.prefix)
    #     self.init_prefix = self.init_postfix if self.init_postfix is not None else ""
    #     self.repartition_name = self.experiment
    #     self.training_data.training_params.repartition_name = self.repartition_name
    #     self.experiment_name = experiment_name
    #     self.parent_folder_path = os.path.join(
    #         Folders.MODELS_EXPERIMENTS.value, self.experiment_name
    #     )
    #     create_folder(self.parent_folder_path)

    #     initial_folder = os.path.join(Folders.MODELS_AMF_GD_YOLOV8.value, old_model_name)
    #     shutil.copytree(initial_folder, self.folder_path)

    #     if experiment_name != "":
    #         self.parent_folder_path = os.path.join(
    #             Folders.MODELS_EXPERIMENTS.value, experiment_name
    #         )
    #     else:
    #         self.parent_folder_path = Folders.MODELS_AMF_GD_YOLOV8.value

    # def update_remove(self) -> None:
    #     to_remove = [
    #         self.postfix,
    #         self.init_postfix,
    #         self.experiment,
    #         self.training_data.dataset_params.split_random_seed,
    #     ]
    #     for elem in to_remove:
    #         del elem


class ParamsCombinations:

    def __init__(
        self,
        name: str,
        params_dict: Optional[Dict[str, List]] = None,
        forget_combinations: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
    ) -> None:

        self.name = name

        if os.path.exists(self.state_path):
            self.load_state()
            return

        if params_dict is None or forget_combinations is None:
            raise Exception(
                "If a ParamsCombinations object with the same name doesn't exist, every parameter should be specified."
            )

        params_dict["experiment_name"] = [self.name]

        keys, values = zip(*params_dict.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        self.combinations = [
            comb
            for comb in combinations
            if not any([forget_comb(comb) for forget_comb in forget_combinations])
        ]

        self.model_names = [""] * len(self.combinations)
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
    def folder_path(self) -> str:
        folder_path = os.path.join(Folders.MODELS_EXPERIMENTS.value, self.name)
        create_folder(folder_path)
        return folder_path

    @property
    def state_path(self) -> str:
        state_path = os.path.join(self.folder_path, "parameters_combinations.json")
        return state_path

    def __iter__(self):
        total_combinations = len(self.combinations)
        for idx in range(self.next_idx, len(self.combinations)):
            str_len = len(str(total_combinations))
            RICH_PRINTING.print(f"Experiment {idx+1:>{str_len}}/{total_combinations}")
            RICH_PRINTING.print(f"Parameters: {self.combinations[idx]}")

            model_training_session = ModelTrainingSession(**self.combinations[idx])
            yield model_training_session

            self.model_names[idx] = model_training_session.model_name
            self.next_idx += 1
            self.save_state()


def main():
    # install(show_locals=True)
    create_all_folders()

    params_dict = {
        "epochs": [1000],
        "repartition_name": ["exp1", "exp2", "exp3", "exp4"],
        "lr": [6e-3, 2.5e-3, 1e-3],
        "accumulate": [12, 24, 36],
        "proba_drop_rgb": [0, 0.1, 0.333],
        "proba_drop_chm": [0, 0.1, 0.333],
        "model_size": ["n"],
        "agnostic": [True],
        "use_rgb": [True],
        "use_cir": [True],
        "use_chm": [True],
    }

    forget_combinations = [
        lambda d: not d["use_rgb"] and not d["use_cir"] and not d["use_chm"],
        lambda d: d["proba_drop_rgb"] != d["proba_drop_chm"],
    ]

    params_combinations = ParamsCombinations(
        "training_params_experiment",
        params_dict=params_dict,
        forget_combinations=forget_combinations,
    )

    for model_training_session in params_combinations:
        # Training session
        model_training_session.train()

    # experiment_name = "training_params_experiment"
    # parent_folder_path = os.path.join(Folders.MODELS_EXPERIMENTS.value, experiment_name)
    # create_folder(parent_folder_path)
    # shutil.copyfile(
    #     "/home/alexandre/Documents/projects/Geodan_internship/tree-segmentation/models/experiments/training_params_experiment.json",
    #     "/home/alexandre/Documents/projects/Geodan_internship/tree-segmentation/models/experiments/training_params_experiment/parameters_combinations.json",
    # )
    # model_names = os.listdir(Folders.MODELS_AMF_GD_YOLOV8.value)
    # for model_name in model_names:
    #     model_training_session = ModelTrainingSession.from_name(
    #         Folders.MODELS_AMF_GD_YOLOV8.value, model_name
    #     )
    #     model_training_session.update(experiment_name)
    #     model_training_session.update_remove()
    #     model_training_session.save_init_params()
    #     model_training_session.save_params()
    #     model_training_session._save_pickle()

    #     new_model_name = model_training_session.model_name
    #     # print(
    #     #     f"{ModelSession.get_pickle_path(parent_folder_path=parent_folder_path, model_name=model_name) = }"
    #     # )

    #     model_training_session = ModelTrainingSession.from_name(
    #         parent_folder_path=parent_folder_path, model_name=new_model_name
    #     )

    #     file_path = "/home/alexandre/Documents/projects/Geodan_internship/tree-segmentation/models/experiments/training_params_experiment/parameters_combinations.json"
    #     with open(file_path) as f:
    #         data = json.load(f)

    #     for d in data["parameters"]:
    #         if d["model_name"] == model_name:
    #             d["model_name"] = new_model_name

    #     with open(file_path, "w") as f:
    #         json.dump(data, f)

    # model_training_session = ModelTrainingSession.from_name(
    #     parent_folder_path=parent_folder_path, model_name="exp0_5ddfced3391e48b4bba91caf8d4602ef"
    # )
    # model_training_session.compute_metrics()


if __name__ == "__main__":
    main()
    RICH_PRINTING.close()
