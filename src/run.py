from __future__ import annotations

import json
import os
import pickle
from itertools import product
from typing import Optional, Sequence, Tuple

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
        model_size: str = "n",
        lr: float = 1e-2,
        epochs: int = 1000,
        batch_size: int = 10,
        num_workers: int = 0,
        accumulate: int = 10,
        no_improvement_stop_epochs: int = 50,
        proba_drop_rgb: float = 1 / 3,
        proba_drop_chm: float = 1 / 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        postfix: Optional[str] = None,
    ) -> None:
        self.args = locals()

        if chm_z_layers is None:
            z_tops = [1, 2, 3, 5, 7, 10, 15, 20, np.inf]
            chm_z_layers = [(-np.inf, z_top) for z_top in z_tops]

        dataset_params = DatasetParams(
            annotations_file_name=annotations_file_name,
            use_rgb=use_rgb,
            use_cir=use_cir,
            use_chm=use_chm,
            chm_z_layers=chm_z_layers,
            agnostic=agnostic,
        )
        training_params = TrainingParams(
            model_size=model_size,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            accumulate=accumulate,
            no_improvement_stop_epochs=no_improvement_stop_epochs,
            proba_drop_rgb=proba_drop_rgb,
            proba_drop_chm=proba_drop_chm,
        )
        training_data = TrainingData(dataset_params=dataset_params, training_params=training_params)
        super().__init__(training_data=training_data, device=device, postfix=postfix)

    @property
    def init_params_path(self) -> str:
        return os.path.join(self.folder_path, "model_init_params.json")

    def save_init_params(self) -> None:
        save_path = self.init_params_path
        with open(save_path, "w") as fp:
            json.dump(self.__dict__, fp, cls=FullJsonEncoder, sort_keys=True, indent=4)

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
        "model_size": ["n", "s"],
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

    # for combination in filtered_combinations:
    #     # Training session
    #     model_training_session = ModelTrainingSession(**combination)
    #     model_training_session.train()
    #     model_training_session.close()

    # model_training_session = ModelTrainingSession(epochs=0)
    # model_training_session.train()
    # model_training_session.close()

    model_training_session = ModelTrainingSession.from_name("trained_model_1000ep_0")
    model_training_session.save_init_params()
    # model_training_session.compute_metrics()
    model_training_session.close()


if __name__ == "__main__":
    main()
    RICH_PRINTING.close()
