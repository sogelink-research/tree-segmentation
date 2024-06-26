import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeVar, cast

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from box_cls import Box, compute_iou
from dataloaders import convert_ground_truth_from_tensors
from layers import AMF_GD_YOLOv8


List_int_or_str = TypeVar("List_int_or_str", List[int], List[str])
Match = Tuple[Tuple[int, int], float, int | str]


def hungarian_algorithm(
    pred_bboxes: List[Box],
    pred_labels: List_int_or_str,
    gt_bboxes: List[Box],
    gt_labels: List_int_or_str,
    iou_threshold: float,
    agnostic: bool,
) -> Tuple[List[Match], List[Tuple[int, int | str]], List[Tuple[int, int | str]]]:
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
        ((pred, gt), float(1 - cost_matrix[pred, gt]), pred_labels[pred])
        for pred, gt in zip(pred_ind, gt_ind)
        if pred < pred_len and gt < gt_len and cost_matrix[pred, gt] != max_cost
    ]
    matched_pairs.sort(key=lambda t: t[1])

    unmatched_pred = list(
        set(range(pred_len)).difference(set(map(lambda t: t[0][0], matched_pairs)))
    )
    unmatched_pred = [(pred, pred_labels[pred]) for pred in unmatched_pred]
    unmatched_gt = list(set(range(gt_len)).difference(set(map(lambda t: t[0][1], matched_pairs))))
    unmatched_gt = [(gt, gt_labels[gt]) for gt in unmatched_gt]

    return matched_pairs, unmatched_pred, unmatched_gt


def hungarian_algorithm_confs(
    pred_bboxes: List[Box],
    pred_labels: List_int_or_str,
    pred_scores: List[float],
    gt_bboxes: List[Box],
    gt_labels: List_int_or_str,
    iou_threshold: float,
    conf_threshold_list: List[float],
    agnostic: bool,
) -> Tuple[List[List[Match]], List[List[Tuple[int, int | str]]], List[List[Tuple[int, int | str]]]]:
    matched_pairs_list: List[List[Match]] = []
    unmatched_pred_list: List[List[Tuple[int, int | str]]] = []
    unmatched_gt_list: List[List[Tuple[int, int | str]]] = []

    for conf_threshold in conf_threshold_list:
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
    matched_pairs: List[Match],
    unmatched_pred: List[Tuple[int, int | str]],
    unmatched_gt: List[Tuple[int, int | str]],
) -> Tuple[List[float], List[float], float]:
    matched_pairs.sort(key=lambda t: t[1])
    sorted_ious = list(map(lambda t: t[1], matched_pairs))

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


def compute_sorted_ap_per_label(
    matched_pairs: List[Match],
    unmatched_preds: List[Tuple[int, int | str]],
    unmatched_gts: List[Tuple[int, int | str]],
) -> Dict[int | str, Tuple[List[float], List[float], float]]:

    grouped_matched_pairs = defaultdict(list)
    grouped_unmatched_preds = defaultdict(list)
    grouped_unmatched_gts = defaultdict(list)

    for matched_pair in matched_pairs:
        label = matched_pair[2]
        grouped_matched_pairs[label].append(matched_pair)

    for unmatched_pred in unmatched_preds:
        label = unmatched_pred[1]
        grouped_unmatched_preds[label].append(unmatched_pred)

    for unmatched_gt in unmatched_gts:
        label = unmatched_gt[1]
        grouped_unmatched_gts[label].append(unmatched_gt)

    results = {}
    for label in grouped_matched_pairs.keys():
        matched_pairs_temp = grouped_matched_pairs[label]
        unmatched_preds_temp = grouped_unmatched_preds[label]
        unmatched_gts_temp = grouped_unmatched_gts[label]
        results[label] = compute_sorted_ap(
            matched_pairs=matched_pairs_temp,
            unmatched_pred=unmatched_preds_temp,
            unmatched_gt=unmatched_gts_temp,
        )

    return results


def compute_sorted_ap_confs(
    matched_pairs_list: List[List[Match]],
    unmatched_preds_list: List[List[Tuple[int, int | str]]],
    unmatched_gts_list: List[List[Tuple[int, int | str]]],
) -> Tuple[List[List[float]], List[List[float]], List[float]]:

    sorted_ious_list: List[List[float]] = []
    aps_list: List[List[float]] = []
    sorted_ap_list: List[float] = []

    for matched_pairs, unmatched_preds, unmatched_gts in zip(
        matched_pairs_list, unmatched_preds_list, unmatched_gts_list
    ):
        sorted_ious, aps, sorted_ap = compute_sorted_ap(
            matched_pairs, unmatched_preds, unmatched_gts
        )
        sorted_ious_list.append(sorted_ious)
        aps_list.append(aps)
        sorted_ap_list.append(sorted_ap)

    return sorted_ious_list, aps_list, sorted_ap_list


def compute_sorted_ap_confs_per_label(
    matched_pairs_list: List[List[Match]],
    unmatched_preds_list: List[List[Tuple[int, int | str]]],
    unmatched_gts_list: List[List[Tuple[int, int | str]]],
) -> Dict[int | str, Tuple[List[List[float]], List[List[float]], List[float]]]:

    sorted_ious_list_dict: Dict[int | str, List[List[float]]] = {}
    aps_list_dict: Dict[int | str, List[List[float]]] = {}
    sorted_ap_list_dict: Dict[int | str, List[float]] = {}

    for idx, (matched_pairs, unmatched_preds, unmatched_gts) in enumerate(
        zip(matched_pairs_list, unmatched_preds_list, unmatched_gts_list)
    ):
        sorted_ap_per_label = compute_sorted_ap_per_label(
            matched_pairs, unmatched_preds, unmatched_gts
        )
        for label, (sorted_ious, aps, sorted_ap) in sorted_ap_per_label.items():
            sorted_ious_list_dict[label][idx] = sorted_ious
            aps_list_dict[label][idx] = aps
            sorted_ap_list_dict[label][idx] = sorted_ap

    results = {}
    for label in sorted_ap_per_label.keys():
        sorted_ious_list = sorted_ious_list_dict[label]
        aps_list = aps_list_dict[label]
        sorted_ap_list = sorted_ap_list_dict[label]

        results[label] = (sorted_ious_list, aps_list, sorted_ap_list)

    return results


def plot_ap_iou(
    sorted_ious_list: List[List[float]],
    aps_list: List[List[float]],
    sorted_ap_list: List[float],
    conf_threshold_list: List[float],
    legend_list: List[str],
    title: Optional[str] = None,
    show: bool = False,
    save_path: str | None = None,
):
    plt.figure(figsize=(10, 6))

    for sorted_ious, aps, sorted_ap, legend, conf_threshold in zip(
        sorted_ious_list, aps_list, sorted_ap_list, legend_list, conf_threshold_list
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

    title = "Sorted AP curve" if title is None else title
    plt.title(title)
    plt.tight_layout()

    has_legend, _ = plt.gca().get_legend_handles_labels()
    if any(label != "" for label in has_legend):
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


def plot_sap_conf(
    sorted_ap_lists: List[List[float]],
    conf_threshold_lists: List[List[float]],
    legend_list: List[str],
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(10, 6))

    for sorted_ap, conf_threshold, legend in zip(
        sorted_ap_lists, conf_threshold_lists, legend_list
    ):
        x = conf_threshold
        y = sorted_ap

        plt.plot(x, y, label=f"{legend}")
        plt.grid(alpha=0.5)
        plt.xlabel("Confidence threshold")
        plt.ylabel("Sorted AP")

    title = "Sorted AP w.r.t the confidence threshold" if title is None else title
    plt.title(title)
    plt.tight_layout()

    has_legend, _ = plt.gca().get_legend_handles_labels()
    if any(label != "" for label in has_legend):
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


class AP_Metrics:
    def __init__(self, conf_threshold_list: List[float], agnostic: bool = False) -> None:
        self.conf_threshold_list = conf_threshold_list
        self.agnostic = agnostic

        self.reset()

    def reset(self) -> None:
        self.matched_pairs: List[List[Match]] = [[] for _ in range(len(self.conf_threshold_list))]
        self.unmatched_pred: List[List[Tuple[int, int | str]]] = [
            [] for _ in range(len(self.conf_threshold_list))
        ]
        self.unmatched_gt: List[List[Tuple[int, int | str]]] = [
            [] for _ in range(len(self.conf_threshold_list))
        ]

        self.sorted_ap_updated = False
        self.sorted_ap_per_label_updated = False

    def add_preds(
        self,
        model: AMF_GD_YOLOv8,
        preds: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_indices: torch.Tensor,
        image_indices: torch.Tensor,
    ) -> None:
        lowest_conf_threshold = min(self.conf_threshold_list)

        bboxes_list, scores_list, classes_as_ints_list = model.predict_from_preds(
            preds, conf_threshold=lowest_conf_threshold
        )

        gt_bboxes_per_image, gt_classes_per_image = convert_ground_truth_from_tensors(
            gt_bboxes=gt_bboxes,
            gt_classes=gt_classes,
            gt_indices=gt_indices,
            image_indices=image_indices,
        )

        # Compute the matchings for each individual image
        iou_threshold = 1e-6
        for bboxes, scores, classes_as_ints, gt_bboxes_list, gt_classes_list in zip(
            bboxes_list,
            scores_list,
            classes_as_ints_list,
            gt_bboxes_per_image,
            gt_classes_per_image,
        ):
            matched_pairs_temp, unmatched_pred_temp, unmatched_gt_temp = hungarian_algorithm_confs(
                pred_bboxes=bboxes,
                pred_labels=classes_as_ints,
                pred_scores=scores,
                gt_bboxes=gt_bboxes_list,
                gt_labels=gt_classes_list,
                iou_threshold=iou_threshold,
                conf_threshold_list=self.conf_threshold_list,
                agnostic=self.agnostic,
            )
            for i in range(len(self.conf_threshold_list)):
                self.matched_pairs[i].extend(matched_pairs_temp[i])
                self.unmatched_pred[i].extend(unmatched_pred_temp[i])
                self.unmatched_gt[i].extend(unmatched_gt_temp[i])

        self.sorted_ap_updated = False
        self.sorted_ap_per_label_updated = False

    def compute_sorted_ap(self, per_label: bool = False) -> None:
        if not per_label and not self.sorted_ap_updated:
            self.sorted_ious_list, self.aps_list, self.sorted_ap_list = compute_sorted_ap_confs(
                self.matched_pairs, self.unmatched_pred, self.unmatched_gt
            )
            self.sorted_ap_updated = True

        if per_label and not self.sorted_ap_per_label_updated:
            self.sorted_ap_per_label_dict = compute_sorted_ap_confs_per_label(
                self.matched_pairs, self.unmatched_pred, self.unmatched_gt
            )

    def get_sorted_aps(
        self,
    ) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
        """Returns the data of the sortedAP for all the confidence thresholds given to the class.

        Returns:
            Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
            (sorted_ious_list, aps_list, sorted_ap_list, conf_threshold_list) where the first
            dimension of each list corresponds to one confidence threshold value:
            - sorted_ious_list contains the lists of IoUs between the matched pairs, in increasing
            order, for all threshold values.
            - aps_list contains the lists of AP scores corresponding to the IoUs in
            sorted_ious_list.
            - sorted_ap_list contains the sortedAP metrics (the integral of aps w.r.t.
            sorted_ious).
            - conf_threshold_list contains the model confidence thresholds used to compute the
            metrics above.
        """
        self.compute_sorted_ap()
        return self.sorted_ious_list, self.aps_list, self.sorted_ap_list, self.conf_threshold_list

    def get_best_sorted_ap(self) -> Tuple[List[float], List[float], float, float]:
        """Returns the best sortedAP among those computed with the confidence thresholds given to
        the class. The methods also outputs the values of the AP/IoU curve and the corresponding
        threshold.

        Returns:
            Tuple[List[float], List[float], float, float]: (sorted_ious, aps, sorted_ap,
            conf_threshold) where:
            - sorted_ious is the list of IoUs between the matched pairs, in increasing order.
            - aps is the list of AP scores corresponding to the IoUs in sorted_ious.
            - sorted_ap is the sortedAP metrics (the integral of aps w.r.t. sorted_ious).
            - conf_threshold is the corresponding model confidence threshold.
        """
        self.compute_sorted_ap()
        best_index = self.sorted_ap_list.index(max(self.sorted_ap_list))
        best_sorted_ious = self.sorted_ious_list[best_index]
        best_aps = self.aps_list[best_index]
        best_sorted_ap = self.sorted_ap_list[best_index]
        best_conf_threshold = self.conf_threshold_list[best_index]
        return best_sorted_ious, best_aps, best_sorted_ap, best_conf_threshold

    def get_sorted_aps_per_label(
        self,
    ) -> Dict[int | str, Tuple[List[List[float]], List[List[float]], List[float]]]:
        self.compute_sorted_ap(per_label=True)
        return self.sorted_ap_per_label_dict

    def get_best_sorted_ap_per_label(
        self,
    ) -> Dict[int | str, Tuple[List[float], List[float], float, float]]:
        self.compute_sorted_ap(per_label=True)
        best_sorted_ap_dict = {}
        for label, sorted_ap_elems in self.sorted_ap_per_label_dict.items():
            sorted_ious_list, aps_list, sorted_ap_list = sorted_ap_elems

            best_index = sorted_ap_list.index(max(sorted_ap_list))
            best_sorted_ious = sorted_ious_list[best_index]
            best_aps = aps_list[best_index]
            best_sorted_ap = sorted_ap_list[best_index]
            best_conf_threshold = self.conf_threshold_list[best_index]

            best_sorted_ap_dict[label] = (
                best_sorted_ious,
                best_aps,
                best_sorted_ap,
                best_conf_threshold,
            )

        return best_sorted_ap_dict

    def plot_ap_iou_per_label(self, save_path: str, title: Optional[str] = None) -> None:
        best_sorted_ious_list = []
        best_aps_list = []
        best_sorted_ap_list = []
        best_conf_threshold_list = []
        legend_list = []

        best_sorted_ap_dict = self.get_best_sorted_ap_per_label()

        for label, (
            best_sorted_ious,
            best_aps,
            best_sorted_ap,
            best_conf_threshold,
        ) in best_sorted_ap_dict.items():
            legend_list.append(label)
            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_conf_threshold_list.append(best_conf_threshold)

        plot_ap_iou(
            sorted_ious_list=best_sorted_ious_list,
            aps_list=best_aps_list,
            sorted_ap_list=best_sorted_ap_list,
            conf_threshold_list=best_conf_threshold_list,
            legend_list=legend_list,
            title=title,
            save_path=save_path,
        )

    def plot_sap_conf_per_label(self, save_path: str, title: Optional[str] = None) -> None:
        sorted_ap_lists = []
        conf_threshold_lists = []
        legend_list = []

        sorted_aps_per_label_dict = self.get_sorted_aps_per_label()

        for label, (
            sorted_ious_list,
            aps_list,
            sorted_ap_list,
        ) in sorted_aps_per_label_dict.items():
            sorted_ap_lists.append(sorted_ap_list)
            conf_threshold_lists.append(self.conf_threshold_list)
            legend_list.append(label)

        plot_sap_conf(
            sorted_ap_lists=sorted_ap_lists,
            conf_threshold_lists=conf_threshold_lists,
            legend_list=legend_list,
            title=title,
            save_path=save_path,
        )


class AP_Metrics_List:
    def __init__(
        self,
        ap_metrics_list: Optional[List[AP_Metrics]] = None,
        legend_list: Optional[List[str]] = None,
    ) -> None:
        if (ap_metrics_list is None and legend_list is not None) or (
            ap_metrics_list is not None and legend_list is None
        ):
            raise ValueError(
                "You should either specify ap_metrics_list and legend_list or neither."
            )
        if (ap_metrics_list is not None and legend_list is not None) and (
            len(ap_metrics_list) != len(legend_list)
        ):
            raise ValueError("ap_metrics_list and legend_list should have the same length.")
        self.ap_metrics_list = [] if ap_metrics_list is None else ap_metrics_list
        self.legend_list = [] if legend_list is None else legend_list

    def add_ap_metrics(self, ap_metrics: AP_Metrics, legend) -> None:
        ap_metrics.compute_sorted_ap()
        self.ap_metrics_list.append(ap_metrics)
        self.legend_list.append(legend)

    def plot_ap_iou(self, save_path: str, title: Optional[str] = None) -> None:
        best_sorted_ious_list = []
        best_aps_list = []
        best_sorted_ap_list = []
        best_conf_threshold_list = []

        for ap_metrics in self.ap_metrics_list:
            best_sorted_ious, best_aps, best_sorted_ap, best_conf_threshold = (
                ap_metrics.get_best_sorted_ap()
            )
            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_conf_threshold_list.append(best_conf_threshold)

        plot_ap_iou(
            sorted_ious_list=best_sorted_ious_list,
            aps_list=best_aps_list,
            sorted_ap_list=best_sorted_ap_list,
            conf_threshold_list=best_conf_threshold_list,
            legend_list=self.legend_list,
            title=title,
            save_path=save_path,
        )

    def plot_sap_conf(self, save_path: str, title: Optional[str] = None) -> None:
        sorted_ap_lists = []
        conf_threshold_lists = []

        for ap_metrics in self.ap_metrics_list:
            ap_metrics.compute_sorted_ap()
            sorted_ap_lists.append(ap_metrics.sorted_ap_list)
            conf_threshold_lists.append(ap_metrics.conf_threshold_list)

        plot_sap_conf(
            sorted_ap_lists=sorted_ap_lists,
            conf_threshold_lists=conf_threshold_lists,
            legend_list=self.legend_list,
            title=title,
            save_path=save_path,
        )
