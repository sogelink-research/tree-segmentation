from typing import Sequence

import numpy as np
import torch

from model_session import DatasetParams, ModelSession, TrainingData, TrainingParams


def main():
    # Data parameters
    annotations_file_name = "122000_484000.geojson"
    z_tops: Sequence[float] = [1, 2, 3, 5, 7, 10, 15, 20, np.inf]
    z_limits_list = [(-np.inf, z_top) for z_top in z_tops]

    dataset_params = DatasetParams(
        annotations_file_name=annotations_file_name,
        use_rgb=True,
        use_cir=True,
        use_chm=True,
        chm_z_layers=z_limits_list,
    )

    # Training parameters

    lr = 1e-2
    epochs = 0
    batch_size = 10
    num_workers = 0
    accumulate = 10

    proba_drop_rgb = 1 / 3
    proba_drop_chm = 1 / 3

    postfix = "rgb_cir_multi_chm"

    training_params = TrainingParams(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        accumulate=accumulate,
        proba_drop_rgb=proba_drop_rgb,
        proba_drop_chm=proba_drop_chm,
    )

    # Training data
    training_data = TrainingData(dataset_params=dataset_params, training_params=training_params)

    # Training session
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_session = ModelSession(training_data=training_data, device=device, postfix=postfix)

    model_session.train()

    model_session.close()

    # model_session = ModelSession.from_name(
    #     "trained_model_rgb_cir_multi_chm_1500ep_2", device=device
    # )
    # model_session.compute_metrics()


if __name__ == "__main__":
    main()
