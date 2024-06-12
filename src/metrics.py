import itertools
from typing import List, Tuple, TypeVar, cast

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from skimage import io

from box_cls import Box, compute_iou
from plot import create_bboxes_image


List_int_or_str = TypeVar("List_int_or_str", List[int], List[str])


def hungarian_algorithm(
    pred_bboxes: List[Box],
    pred_labels: List_int_or_str,
    gt_bboxes: List[Box],
    gt_labels: List_int_or_str,
    iou_threshold: float,
    agnostic: bool,
) -> Tuple[List[Tuple[Tuple[int, int], float]], List[int], List[int]]:
    pred_len = len(pred_bboxes)
    gt_len = len(gt_bboxes)

    # Large values replacing inf
    max_cost = 1e6
    cost_size = max(pred_len, gt_len)
    cost_matrix = np.full((cost_size, cost_size), float(max_cost))

    for i, (pred_box, pred_class) in enumerate(zip(pred_bboxes, pred_labels)):
        for j, (gt_box, gt_class) in enumerate(zip(gt_bboxes, gt_labels)):
            iou = compute_iou(pred_box, gt_box)
            if iou > iou_threshold:
                if agnostic or pred_class == gt_class:
                    cost_matrix[i, j] = 1 - iou

    # Compute the matching
    pred_ind, gt_ind = linear_sum_assignment(cost_matrix)

    # Extract pairs of indices where matches were found
    matched_pairs = [
        ((pred, gt), float(cost_matrix[pred, gt]))
        for pred, gt in zip(pred_ind, gt_ind)
        if pred < pred_len and gt < gt_len and cost_matrix[pred, gt] != max_cost
    ]

    unmatched_pred = list(
        set(range(pred_len)).difference(set(map(lambda t: t[0][0], matched_pairs)))
    )
    unmatched_gt = list(set(range(gt_len)).difference(set(map(lambda t: t[0][1], matched_pairs))))

    return matched_pairs, unmatched_pred, unmatched_gt


def hungarian_algorithm_confs(
    pred_bboxes: List[Box],
    pred_labels: List_int_or_str,
    pred_scores: List[float],
    gt_bboxes: List[Box],
    gt_labels: List_int_or_str,
    iou_threshold: float,
    conf_thresholds: List[float],
    agnostic: bool,
) -> Tuple[List[List[Tuple[Tuple[int, int], float]]], List[List[int]], List[List[int]]]:
    matched_pairs_list: List[List[Tuple[Tuple[int, int], float]]] = []
    unmatched_pred_list: List[List[int]] = []
    unmatched_gt_list: List[List[int]] = []

    for conf_threshold in conf_thresholds:
        mask = [i for i in range(len(pred_scores)) if pred_scores[i] > conf_threshold]
        pred_bboxes_conf = [pred_bboxes[i] for i in mask]
        pred_labels_conf = cast(List_int_or_str, [pred_labels[i] for i in mask])
        matched_pairs, unmatched_pred, unmatched_gt = hungarian_algorithm(
            pred_bboxes_conf, pred_labels_conf, gt_bboxes, gt_labels, iou_threshold, agnostic
        )
        matched_pairs_list.append(matched_pairs)
        unmatched_pred_list.append(unmatched_pred)
        unmatched_gt_list.append(unmatched_gt)

    return matched_pairs_list, unmatched_pred_list, unmatched_gt_list


def compute_sorted_ap(
    matched_pairs: List[Tuple[Tuple[int, int], float]],
    unmatched_pred: List[int],
    unmatched_gt: List[int],
) -> Tuple[List[float], List[float], float]:
    sorted_ious = sorted(map(lambda t: t[1], matched_pairs))

    tp0 = len(matched_pairs)
    fp0 = len(unmatched_pred)
    p = tp0 + fp0
    fn0 = len(unmatched_gt)

    if tp0 == 0:
        return [], [], 0.0
    aps = [(tp0) / (p + fn0)]
    sorted_ap = sorted_ious[0] * aps[0]
    for k in range(1, tp0):
        aps.append((tp0 - k) / (p + fn0 + k))
        sorted_ap += 0.5 * (sorted_ious[k] - sorted_ious[k - 1]) * (aps[k] + aps[k - 1])
    return sorted_ious, aps, sorted_ap


def compute_sorted_ap_confs(
    matched_pairs_list: List[List[Tuple[Tuple[int, int], float]]],
    unmatched_pred_list: List[List[int]],
    unmatched_gt_list: List[List[int]],
) -> Tuple[List[List[float]], List[List[float]], List[float]]:

    sorted_ious_list: List[List[float]] = []
    aps_list: List[List[float]] = []
    sorted_ap_list: List[float] = []

    for matched_pairs, unmatched_pred, unmatched_gt in zip(
        matched_pairs_list, unmatched_pred_list, unmatched_gt_list
    ):
        sorted_ious, aps, sorted_ap = compute_sorted_ap(matched_pairs, unmatched_pred, unmatched_gt)
        sorted_ious_list.append(sorted_ious)
        aps_list.append(aps)
        sorted_ap_list.append(sorted_ap)

    return sorted_ious_list, aps_list, sorted_ap_list


def plot_sorted_ap(
    sorted_ious_list: List[List[float]],
    aps_list: List[List[float]],
    sorted_ap_list: List[float],
    legend_list: List[str],
    conf_thresholds: List[float],
    show: bool = False,
    save_path: str | None = None,
):
    plt.figure(1, figsize=(10, 6))
    plt.clf()  # Clear the current figure

    for sorted_ious, aps, sorted_ap, legend, conf_threshold in zip(
        sorted_ious_list, aps_list, sorted_ap_list, legend_list, conf_thresholds
    ):
        x = [0.0]
        y = [aps[0] if len(aps) > 0 else 0.0]
        x.extend(sorted_ious)
        y.extend(aps)
        x.append(1.0)
        y.append(0.0)

        plt.plot(
            x,
            y,
            label=f"{legend}\n- conf_threshold = {round(conf_threshold, 5)}\n- sortedAP = {round(sorted_ap, 4)}",
        )
        plt.grid(alpha=0.5)
        plt.xlabel("IoU")
        plt.ylabel("AP")

    plt.title("Sorted AP")
    plt.tight_layout()

    has_legend, _ = plt.gca().get_legend_handles_labels()
    if any(label != "" for label in has_legend):
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


def plot_sorted_ap_confs(
    sorted_ap_lists: List[List[float]],
    conf_thresholds_list: List[List[float]],
    legend_list: List[str],
    show: bool = False,
    save_path: str | None = None,
):
    plt.figure(1, figsize=(10, 6))
    plt.clf()  # Clear the current figure

    for sorted_ap, conf_threshold, legend in zip(
        sorted_ap_lists, conf_thresholds_list, legend_list
    ):
        x = conf_threshold
        y = sorted_ap

        plt.plot(x, y, label=f"{legend}")
        plt.grid(alpha=0.5)
        plt.xlabel("Confidence threshold")
        plt.ylabel("Sorted AP")

    plt.title("Sorted AP w.r.t the confidence threshold")
    plt.tight_layout()

    has_legend, _ = plt.gca().get_legend_handles_labels()
    if any(label != "" for label in has_legend):
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()
