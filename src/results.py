import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from training import rgb_cir_chm_usage_legend


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


data_folder = "models/amf_gd_yolov8"
df = best_sorted_ap_per_experiment(data_folder)
print(df)

df["Group"] = df.apply(
    lambda row: rgb_cir_chm_usage_legend(row["use_rgb"], row["use_cir"], row["use_chm"]), axis=1
)
df.sort_values(by=["use_rgb", "use_cir", "use_chm"], inplace=True)

print(df["Group"])

plt.figure(figsize=(10, 6))

# Use seaborn to create a scatter plot with different colors for each group
# sns.violinplot(data=df, x="Group", y="Best sortedAP", hue="Group", palette="deep")
sns.scatterplot(
    data=df,
    x="Group",
    y="Best sortedAP",
    hue="Group",
    style="Data used for evaluation",
    s=100,
    palette="deep",
)

# Customizing the plot
plt.xlabel("Data used for evaluation")
plt.ylabel("Best sortedAP")
plt.title("Best sortedAP w.r.t type of data used")
plt.grid(True)
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
