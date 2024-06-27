from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from model_session import DatasetParams, ModelSession, TrainingData, TrainingParams


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


def main():
    params_dict = {
        "agnostic": [True, False],
        "use_rgb": [True, False],
        "use_cir": [True, False],
        "use_chm": [True, False],
    }

    forget_combinations = [
        {
            "use_rgb": False,
            "use_cir": False,
            "use_chm": False,
        },
        {
            "agnostic": True,
            "use_rgb": True,
            "use_cir": True,
            "use_chm": True,
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
        model_training_session.close()

    # model_training_session = ModelSession.from_name("trained_model_1000ep_0")
    # model_training_session.training_data.dataset_params.agnostic = False
    # model_training_session.compute_metrics()
    # model_training_session.close()


if __name__ == "__main__":
    main()
