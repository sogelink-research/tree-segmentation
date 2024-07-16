import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

from training import rgb_cir_chm_usage_legend
from utils import RICH_PRINTING, Folders, create_all_folders


create_all_folders()


def read_params(experiment_folder: str) -> Dict[str, Any]:
    params_filenames = [
        "model_init_params.json",
        "model_params.json",
    ]

    combined_model_params = {}
    for filename in params_filenames:
        file_path = os.path.join(experiment_folder, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            for key in data.keys():
                combined_model_params[key] = [data[key]]
    return combined_model_params


def read_best_sorted_ap(experiment_folder: str) -> Dict[str, Any]:
    ap_results_path = os.path.join(experiment_folder, "ap_results_test-set.json")
    with open(ap_results_path, "r") as file:
        data = json.load(file)
    keys = ["legend_list", "best_sorted_ap_list"]
    new_keys = ["Data used for evaluation", "Best sortedAP"]
    new_dict = {new_key: data[key] for key, new_key in zip(keys, new_keys)}
    return new_dict


def best_sorted_ap_per_experiment(data_folder: str):
    """
    Collect data from all experiment folders and return a DataFrame with one row per experiment.
    """
    all_dfs = []
    experiment_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    for experiment_folder in experiment_folders:
        try:
            params = read_params(experiment_folder)
            df = pd.DataFrame(params)
            best_sorted_ap_dict = read_best_sorted_ap(experiment_folder)
            sorted_ap_length = len(best_sorted_ap_dict["Data used for evaluation"])
            df_repeated = pd.concat([df] * sorted_ap_length, ignore_index=True)
            df_best_sorted_ap = pd.DataFrame(best_sorted_ap_dict)
            df_combined = pd.concat([df_repeated, df_best_sorted_ap], axis=1)
            all_dfs.append(df_combined)
        except Exception:
            pass

    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    return df


def plot_sorted_ap_per_data_type(data_folder: str, show: bool, save_path: Optional[str] = None):
    if not show and save_path is None:
        RICH_PRINTING.print("Nothing to do!")
        return

    df = best_sorted_ap_per_experiment(data_folder)

    df["Group"] = df.apply(
        lambda row: rgb_cir_chm_usage_legend(row["use_rgb"], row["use_cir"], row["use_chm"]), axis=1
    )
    df.sort_values(by=["use_rgb", "use_cir", "use_chm"], inplace=True)

    # Create a mapping from Group to numerical values
    group_mapping = {group: idx for idx, group in enumerate(df["Group"].unique())}
    df["Numerical Group"] = df["Group"].map(group_mapping)

    # Define a function to apply random jitter
    def add_jitter(value, jitter_amount=0.1):
        return value + np.random.uniform(-jitter_amount, jitter_amount)

    # Apply random jitter to the numerical group values
    df["Jittered Group"] = df["Numerical Group"].apply(add_jitter)

    plt.figure(figsize=(10, 6))

    # Use seaborn to create a scatter plot with different colors for each group
    # sns.violinplot(data=df, x="Group", y="Best sortedAP", hue="Group", palette="deep")
    # sns.scatterplot(
    #     data=df,
    #     x="Jittered Group",
    #     y="Best sortedAP",
    #     # hue="Group",
    #     hue="Data used for evaluation",
    #     # style="Data used for evaluation",
    #     marker="x",
    #     s=100,
    #     linewidths=2,
    #     palette="deep",
    # )

    sns.swarmplot(data=df, x="Numerical Group", y="Best sortedAP", hue="Data used for evaluation")

    # Customizing the plot
    plt.xlabel("Data used for evaluation")
    plt.ylabel("Best sortedAP")
    plt.title("Best sortedAP w.r.t type of data used")
    plt.grid(True)
    plt.xticks(
        ticks=np.arange(len(df["Group"].unique())), labels=list(df["Group"].unique())
    )  # Ensure original group labels are displayed
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)


def read_training_loss(experiment_folder: str) -> Dict[str, Dict[str, Dict[str, List]]]:
    training_loss_path = os.path.join(experiment_folder, "metrics_values.json")
    with open(training_loss_path, "r") as file:
        data = json.load(file)
    keys = ["Total Loss", "Box Loss", "Class Loss", "Dual Focal Loss"]
    new_keys = ["Total Loss", "Box Loss", "Class Loss", "Dual Focal Loss"]
    new_dict = {new_key: data[key] for key, new_key in zip(keys, new_keys)}
    return new_dict


def compute_instability(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    array = np.array(values)
    mean = np.mean(array)
    variations = np.abs(array[1:] - array[:-1])
    return float(np.mean(variations) / mean)


def compute_training_loss_instability(experiment_folder: str) -> Dict[str, Dict[str, float]]:
    instability_dict = {}
    training_loss_dict = read_training_loss(experiment_folder)
    for loss_name, loss_dict in training_loss_dict.items():
        instability_dict[loss_name] = {}
        for dataset_name, values_dict in loss_dict.items():
            instability_dict[loss_name][dataset_name] = compute_instability(values_dict["avgs"])
    return instability_dict


def training_instability_per_experiment(data_folder: str):
    all_dfs = []
    experiment_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    for experiment_folder in experiment_folders:
        try:
            params = read_params(experiment_folder)
            df = pd.DataFrame(params)
            instability = compute_training_loss_instability(experiment_folder)
            instability_length = 2 * len(instability)
            df_repeated = pd.concat([df] * instability_length, ignore_index=True)
            flattened_instability = [
                (loss_name, dataset_name, value)
                for loss_name, loss_dict in instability.items()
                for dataset_name, value in loss_dict.items()
            ]
            df_instability = pd.DataFrame(
                flattened_instability, columns=["Loss Name", "Data", "Loss instability"]
            )
            df_combined = pd.concat([df_repeated, df_instability], axis=1)
            all_dfs.append(df_combined)
        except Exception:
            pass

    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    return df


def plot_training_instability_per_learning_rate(
    data_folder: str, show: bool, save_path: Optional[str] = None
):
    if not show and save_path is None:
        RICH_PRINTING.print("Nothing to do!")
        return

    df = training_instability_per_experiment(data_folder)
    df = df.groupby(["lr", "Loss Name"])["Loss instability"].apply(list).reset_index()

    # Plotting
    fig, ax = plt.subplots(1, len(df["lr"].unique()), figsize=(15, 5), sharey=True)

    for i, lr in enumerate(df["lr"].unique()):
        subset = df[df["lr"] == lr]
        ax[i].boxplot(subset["Loss instability"], positions=range(len(subset)), patch_artist=True)
        ax[i].set_xticks(range(len(subset)))
        ax[i].set_xticklabels(subset["Loss Name"])
        ax[i].set_title(f"Learning rate = {lr}")
        ax[i].grid(True)

    ax[0].set_ylabel("Loss instability")
    plt.suptitle("Loss instability grouped by learning rate and separated by loss name")
    plt.tight_layout()

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)


def to_tuple(x):
    if isinstance(x, list):
        return tuple(to_tuple(i) for i in x)
    elif isinstance(x, dict):
        return tuple((k, to_tuple(v)) for k, v in sorted(x.items()))
    return x


def all_values_same(column):
    column = column.apply(to_tuple)
    return column.nunique() == 1


data_folder = "models/experiments/training_params_experiment"
best_ap_df = best_sorted_ap_per_experiment(data_folder=data_folder)
print(best_ap_df.columns)
print(best_ap_df)

all_values_same_columns = best_ap_df.apply(all_values_same, axis=0)

# Apply the function to each column and filter columns
filtered_best_ap_df = best_ap_df.drop(
    all_values_same_columns[all_values_same_columns].index, axis=1
)
print(filtered_best_ap_df.columns)
print(filtered_best_ap_df)

filtered_best_ap_df = filtered_best_ap_df.drop(
    columns=[
        "batch_size",
        "postfix",
        "class_names",
        "mean_chm",
        "mean_rgb_cir",
        "model_path",
        "std_chm",
        "std_rgb_cir",
    ],
    errors="ignore",
)

print(filtered_best_ap_df.columns)
print(filtered_best_ap_df)

save_path = os.path.join(Folders.MODELS_RESULTS.value, "training_params_experiment.csv")
filtered_best_ap_df.to_csv(save_path, index=False)


def color_bool(b: bool):
    color = {True: "green", False: "red"}
    return f'background-color: {color.get(b, "")}'


filtered_best_ap_df_style = filtered_best_ap_df.style

# filtered_best_ap_df_style = filtered_best_ap_df_style.map(
#     color_bool, subset=[""]
# )
filtered_best_ap_df_style = filtered_best_ap_df_style.background_gradient(
    subset=["lr", "proba_drop_chm", "proba_drop_rgb", "best_epoch", "Best sortedAP"], cmap="viridis"
)

filtered_best_ap_df.sort_values(
    by=["accumulate", "lr", "proba_drop_chm", "Data used for evaluation"],
    # ascending=[True, False],
    inplace=True,
)

save_path = os.path.join(Folders.MODELS_RESULTS.value, "training_params_experiment.html")
filtered_best_ap_df_style.to_html(save_path, index=False)

filtered_best_ap_df.sort_values(
    by=["Best sortedAP"],
    # ascending=[True, False],
    inplace=True,
)

save_path = os.path.join(Folders.MODELS_RESULTS.value, "training_params_experiment_sorted.html")
filtered_best_ap_df_style.to_html(save_path, index=False)

print(filtered_best_ap_df.columns)
print(filtered_best_ap_df)

# plt.figure()
# # matplotlib.use("Agg")
# sns.pairplot(filtered_best_ap_df, hue="Best sortedAP", palette="Spectral")
# save_path = os.path.join(Folders.MODELS_RESULTS.value, "training_params_experiment_pairplot.png")
# plt.savefig(save_path)

filtered_best_ap_df.sort_values(
    by=["proba_drop_chm", "Data used for evaluation"],
    inplace=True,
)
# filtered_best_ap_df["Group"] = filtered_best_ap_df.apply(
#     lambda row: f'Proba drop: {row["proba_drop_chm"]},\nData:{row["Data used for evaluation"]}',
#     axis=1,
# )
# sns.swarmplot(
#     data=filtered_best_ap_df,
#     x="accumulate",
#     y="Best sortedAP",
#     hue="Group",
#     # dodge=True,
# )
sns.set_style("ticks", {"axes.grid": True})
sns.catplot(
    data=filtered_best_ap_df,
    kind="swarm",
    x="accumulate",
    y="Best sortedAP",
    hue="Data used for evaluation",
    row="lr",
    col="proba_drop_chm",
    margin_titles=True,
    height=2,
    aspect=1,
)

# plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
save_path = os.path.join(Folders.MODELS_RESULTS.value, "training_params_experiment_swarmplot.png")
plt.savefig(save_path, dpi=200)


# ap_per_data_type_path = os.path.join(Folders.MODELS_RESULTS.value, "ap_per_data_type.png")
# plot_sorted_ap_per_data_type(data_folder=data_folder, show=False, save_path=ap_per_data_type_path)
# instability_per_learning_rate_path = os.path.join(
#     Folders.MODELS_RESULTS.value, "instability_per_learning_rate.png"
# )
# plot_training_instability_per_learning_rate(
#     data_folder=data_folder, show=False, save_path=instability_per_learning_rate_path
# )
