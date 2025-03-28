import json
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


Match = Tuple[Tuple[int, int], float, str]


def hungarian_algorithm(
    pred_bboxes: List[Box],
    pred_labels: List[str],
    gt_bboxes: List[Box],
    gt_labels: List[str],
    iou_threshold: float,
    agnostic: bool,
) -> Tuple[List[Match], List[Tuple[int, str]], List[Tuple[int, str]]]:
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
        ((pred, gt), float(1 - cost_matrix[pred, gt]), gt_labels[gt])
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
    pred_labels: List[str],
    pred_scores: List[float],
    gt_bboxes: List[Box],
    gt_labels: List[str],
    iou_threshold: float,
    conf_threshold_list: List[float],
    agnostic: bool,
) -> Tuple[List[List[Match]], List[List[Tuple[int, str]]], List[List[Tuple[int, str]]]]:
    matched_pairs_list: List[List[Match]] = []
    unmatched_pred_list: List[List[Tuple[int, str]]] = []
    unmatched_gt_list: List[List[Tuple[int, str]]] = []

    for conf_threshold in conf_threshold_list:
        mask = [i for i in range(len(pred_scores)) if pred_scores[i] > conf_threshold]
        pred_bboxes_conf = [pred_bboxes[i] for i in mask]
        pred_labels_conf = [pred_labels[i] for i in mask]
        matched_pairs, unmatched_pred, unmatched_gt = hungarian_algorithm(
            pred_bboxes_conf, pred_labels_conf, gt_bboxes, gt_labels, iou_threshold, agnostic
        )
        matched_pairs_list.append(matched_pairs)
        unmatched_pred_list.append(unmatched_pred)
        unmatched_gt_list.append(unmatched_gt)

    return matched_pairs_list, unmatched_pred_list, unmatched_gt_list


def compute_sorted_ap(
    matched_pairs: List[Match],
    unmatched_pred: List[Tuple[int, str]],
    unmatched_gt: List[Tuple[int, str]],
) -> Tuple[List[float], List[float], float, Tuple[int, int]]:
    """_summary_

    Args:
        matched_pairs (List[Match]): _description_
        unmatched_pred (List[Tuple[int, str]]): _description_
        unmatched_gt (List[Tuple[int, str]]): _description_

    Returns:
        Tuple[List[float], List[float], float, Tuple[int, int]]:
        - The IoU values of the matched pairs, in increasing order
        - The corresponding AP values
        - The final sortedAP value
        - The number of predicted and ground-truth boxes
    """
    matched_pairs.sort(key=lambda t: t[1])
    sorted_ious = list(map(lambda t: t[1], matched_pairs))

    tp0 = len(matched_pairs)
    fp0 = len(unmatched_pred)
    p = tp0 + fp0
    fn0 = len(unmatched_gt)

    if tp0 == 0:
        return [], [], 0.0, (tp0 + fp0, tp0 + fn0)
    aps = [(tp0) / (p + fn0)]
    sorted_ap = sorted_ious[0] * aps[0]
    for k in range(1, tp0):
        aps.append((tp0 - k) / (p + fn0 + k))
        sorted_ap += 0.5 * (sorted_ious[k] - sorted_ious[k - 1]) * (aps[k] + aps[k - 1])
    return sorted_ious, aps, sorted_ap, (tp0 + fp0, tp0 + fn0)


def compute_sorted_ap_per_label(
    matched_pairs: List[Match],
    unmatched_preds: List[Tuple[int, str]],
    unmatched_gts: List[Tuple[int, str]],
    class_names: Dict[int, str],
) -> Dict[str, Tuple[List[float], List[float], float, Tuple[int, int]]]:

    grouped_matched_pairs = {label: [] for label in class_names.values()}
    grouped_unmatched_preds = {label: [] for label in class_names.values()}
    grouped_unmatched_gts = {label: [] for label in class_names.values()}

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
    for label in class_names.values():
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
    unmatched_preds_list: List[List[Tuple[int, str]]],
    unmatched_gts_list: List[List[Tuple[int, str]]],
) -> Tuple[List[List[float]], List[List[float]], List[float], List[Tuple[int, int]]]:

    sorted_ious_list: List[List[float]] = []
    aps_list: List[List[float]] = []
    sorted_ap_list: List[float] = []
    boxes_len_list: List[Tuple[int, int]] = []

    for matched_pairs, unmatched_preds, unmatched_gts in zip(
        matched_pairs_list, unmatched_preds_list, unmatched_gts_list
    ):
        sorted_ious, aps, sorted_ap, boxes_len = compute_sorted_ap(
            matched_pairs, unmatched_preds, unmatched_gts
        )
        sorted_ious_list.append(sorted_ious)
        aps_list.append(aps)
        sorted_ap_list.append(sorted_ap)
        boxes_len_list.append(boxes_len)

    return sorted_ious_list, aps_list, sorted_ap_list, boxes_len_list


def compute_sorted_ap_confs_per_label(
    matched_pairs_list: List[List[Match]],
    unmatched_preds_list: List[List[Tuple[int, str]]],
    unmatched_gts_list: List[List[Tuple[int, str]]],
    class_names: Dict[int, str],
) -> Dict[str, Tuple[List[List[float]], List[List[float]], List[float], List[Tuple[int, int]]]]:

    sorted_ious_list_dict: Dict[str, List[List[float]]] = defaultdict(list)
    aps_list_dict: Dict[str, List[List[float]]] = defaultdict(list)
    sorted_ap_list_dict: Dict[str, List[float]] = defaultdict(list)
    boxes_len_list_dict: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for matched_pairs, unmatched_preds, unmatched_gts in zip(
        matched_pairs_list, unmatched_preds_list, unmatched_gts_list
    ):
        sorted_ap_per_label = compute_sorted_ap_per_label(
            matched_pairs, unmatched_preds, unmatched_gts, class_names=class_names
        )
        for label, (sorted_ious, aps, sorted_ap, boxes_len) in sorted_ap_per_label.items():
            sorted_ious_list_dict[label].append(sorted_ious)
            aps_list_dict[label].append(aps)
            sorted_ap_list_dict[label].append(sorted_ap)
            boxes_len_list_dict[label].append(boxes_len)

    results: Dict[
        str, Tuple[List[List[float]], List[List[float]], List[float], List[Tuple[int, int]]]
    ] = {}
    for label in sorted_ap_per_label.keys():
        sorted_ious_list = sorted_ious_list_dict[label]
        aps_list = aps_list_dict[label]
        sorted_ap_list = sorted_ap_list_dict[label]
        boxes_len_list = boxes_len_list_dict[label]

        results[label] = (sorted_ious_list, aps_list, sorted_ap_list, boxes_len_list)

    return results


def plot_ap_iou(
    sorted_ious_list: List[List[float]],
    aps_list: List[List[float]],
    sorted_ap_list: List[float],
    boxes_len_list: List[Tuple[int, int]],
    conf_threshold_list: List[float],
    legend_list: List[str],
    title: Optional[str] = None,
    show: bool = False,
    save_path: str | None = None,
):
    plt.figure(figsize=(10, 6))

    for sorted_ious, aps, sorted_ap, legend, boxes_len, conf_threshold in zip(
        sorted_ious_list, aps_list, sorted_ap_list, legend_list, boxes_len_list, conf_threshold_list
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
            label=f"{legend}\n- Confidence thresh. = {round(conf_threshold, 5)}\n- sortedAP = {round(sorted_ap, 4)}\n- Boxes (pred., gt) = {boxes_len}",
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
    def __init__(
        self, conf_threshold_list: List[float], class_names: Dict[int, str], agnostic: bool = False
    ) -> None:
        self.conf_threshold_list = conf_threshold_list
        self.class_names = class_names
        self.agnostic = agnostic

        self.reset()

    def reset(self) -> None:
        self.matched_pairs: List[List[Match]] = [[] for _ in range(len(self.conf_threshold_list))]
        self.unmatched_pred: List[List[Tuple[int, str]]] = [
            [] for _ in range(len(self.conf_threshold_list))
        ]
        self.unmatched_gt: List[List[Tuple[int, str]]] = [
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
        gt_non_agnostic_classes: torch.Tensor,
        gt_indices: torch.Tensor,
        image_indices: torch.Tensor,
    ) -> None:
        lowest_conf_threshold = min(self.conf_threshold_list)

        bboxes_list, scores_list, classes_as_ints_list = model.predict_from_preds(
            preds, conf_threshold=lowest_conf_threshold
        )

        gt_bboxes_per_image, gt_classes_per_image, gt_non_agnostic_classes_per_image = (
            convert_ground_truth_from_tensors(
                gt_bboxes=gt_bboxes,
                gt_classes=gt_classes,
                gt_non_agnostic_classes=gt_non_agnostic_classes,
                gt_indices=gt_indices,
                image_indices=image_indices,
            )
        )

        # Compute the matchings for each individual image
        iou_threshold = 1e-6
        for bboxes, scores, classes_as_ints, gt_bboxes_list, gt_classes_list in zip(
            bboxes_list,
            scores_list,
            classes_as_ints_list,
            gt_bboxes_per_image,
            gt_non_agnostic_classes_per_image,
        ):
            classes_as_strs = [self.class_names[i] for i in classes_as_ints]
            gt_classes_list_as_strs = [self.class_names[i] for i in gt_classes_list]
            matched_pairs_temp, unmatched_pred_temp, unmatched_gt_temp = hungarian_algorithm_confs(
                pred_bboxes=bboxes,
                pred_labels=classes_as_strs,
                pred_scores=scores,
                gt_bboxes=gt_bboxes_list,
                gt_labels=gt_classes_list_as_strs,
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
            self.sorted_ious_list, self.aps_list, self.sorted_ap_list, self.boxes_len_list = (
                compute_sorted_ap_confs(self.matched_pairs, self.unmatched_pred, self.unmatched_gt)
            )
            self.sorted_ap_updated = True

        elif per_label and not self.sorted_ap_per_label_updated:
            self.sorted_ap_per_label_dict = compute_sorted_ap_confs_per_label(
                self.matched_pairs,
                self.unmatched_pred,
                self.unmatched_gt,
                class_names=self.class_names,
            )

            self.sorted_ap_per_label_updated = True

    def get_sorted_aps(
        self,
    ) -> Tuple[
        List[List[float]], List[List[float]], List[float], List[Tuple[int, int]], List[float]
    ]:
        """Returns the data of the sortedAP for all the confidence thresholds given to the class.

        Returns:
            Tuple[List[List[float]], List[List[float]], List[float], List[Tuple[int, int]], List[float]]:
            (sorted_ious_list, aps_list, sorted_ap_list, conf_threshold_list) where the first
            dimension of each list corresponds to one confidence threshold value:
            - sorted_ious_list contains the lists of IoUs between the matched pairs, in increasing
            order, for all threshold values.
            - aps_list contains the lists of AP scores corresponding to the IoUs in
            sorted_ious_list.
            - sorted_ap_list contains the sortedAP metrics (the integral of aps w.r.t.
            sorted_ious).
            - boxes_len_list contains the total number of boxes.
            - conf_threshold_list contains the model confidence thresholds used to compute the
            metrics above.
        """
        self.compute_sorted_ap()
        return (
            self.sorted_ious_list,
            self.aps_list,
            self.sorted_ap_list,
            self.boxes_len_list,
            self.conf_threshold_list,
        )

    def get_best_sorted_ap(self) -> Tuple[List[float], List[float], float, Tuple[int, int], float]:
        """Returns the best sortedAP among those computed with the confidence thresholds given to
        the class. The methods also outputs the values of the AP/IoU curve and the corresponding
        threshold.

        Returns:
            Tuple[List[float], List[float], float, Tuple[int, int], float]: (sorted_ious, aps,
            sorted_ap, boxes_len, conf_threshold) where:
            - sorted_ious is the list of IoUs between the matched pairs, in increasing order.
            - aps is the list of AP scores corresponding to the IoUs in sorted_ious.
            - sorted_ap is the sortedAP metrics (the integral of aps w.r.t. sorted_ious).
            - boxes_len is the total number of boxes.
            - conf_threshold is the corresponding model confidence threshold.
        """
        self.compute_sorted_ap()
        best_index = self.sorted_ap_list.index(max(self.sorted_ap_list))
        best_sorted_ious = self.sorted_ious_list[best_index]
        best_aps = self.aps_list[best_index]
        best_sorted_ap = self.sorted_ap_list[best_index]
        best_boxes_len = self.boxes_len_list[best_index]
        best_conf_threshold = self.conf_threshold_list[best_index]
        return best_sorted_ious, best_aps, best_sorted_ap, best_boxes_len, best_conf_threshold

    def get_sorted_aps_per_label(
        self,
    ) -> Dict[str, Tuple[List[List[float]], List[List[float]], List[float], List[Tuple[int, int]]]]:
        self.compute_sorted_ap(per_label=True)
        return self.sorted_ap_per_label_dict

    def get_best_sorted_ap_per_label(
        self,
    ) -> Dict[str, Tuple[List[float], List[float], float, Tuple[int, int], float]]:
        self.compute_sorted_ap(per_label=True)
        best_sorted_ap_dict: Dict[
            str, Tuple[List[float], List[float], float, Tuple[int, int], float]
        ] = {}
        for label, sorted_ap_elems in self.sorted_ap_per_label_dict.items():
            sorted_ious_list, aps_list, sorted_ap_list, boxes_len_list = sorted_ap_elems

            best_index = sorted_ap_list.index(max(sorted_ap_list))
            best_sorted_ious = sorted_ious_list[best_index]
            best_aps = aps_list[best_index]
            best_sorted_ap = sorted_ap_list[best_index]
            best_boxes_len = boxes_len_list[best_index]
            best_conf_threshold = self.conf_threshold_list[best_index]

            best_sorted_ap_dict[label] = (
                best_sorted_ious,
                best_aps,
                best_sorted_ap,
                best_boxes_len,
                best_conf_threshold,
            )

        return best_sorted_ap_dict

    def plot_ap_iou_per_label(self, save_path: str, title: Optional[str] = None) -> None:
        best_sorted_ious_list = []
        best_aps_list = []
        best_sorted_ap_list = []
        best_boxes_len_list = []
        best_conf_threshold_list = []
        legend_list = []

        best_sorted_ap_dict = self.get_best_sorted_ap_per_label()

        for label, (
            best_sorted_ious,
            best_aps,
            best_sorted_ap,
            best_boxes_len,
            best_conf_threshold,
        ) in best_sorted_ap_dict.items():
            legend_list.append(label)
            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_boxes_len_list.append(best_boxes_len)
            best_conf_threshold_list.append(best_conf_threshold)

        plot_ap_iou(
            sorted_ious_list=best_sorted_ious_list,
            aps_list=best_aps_list,
            sorted_ap_list=best_sorted_ap_list,
            boxes_len_list=best_boxes_len_list,
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

        print(f"{self.conf_threshold_list = }")

        for label, (
            sorted_ious_list,
            aps_list,
            sorted_ap_list,
            boxes_len_list,
        ) in sorted_aps_per_label_dict.items():
            print(f"{label = }")
            print(f"{sorted_ap_list = }")
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
        best_boxes_len_list = []
        best_conf_threshold_list = []

        for ap_metrics in self.ap_metrics_list:
            best_sorted_ious, best_aps, best_sorted_ap, best_boxes_len, best_conf_threshold = (
                ap_metrics.get_best_sorted_ap()
            )
            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_boxes_len_list.append(best_boxes_len)
            best_conf_threshold_list.append(best_conf_threshold)

        plot_ap_iou(
            sorted_ious_list=best_sorted_ious_list,
            aps_list=best_aps_list,
            sorted_ap_list=best_sorted_ap_list,
            boxes_len_list=best_boxes_len_list,
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

    def save_data(self, save_path: str) -> None:
        best_sorted_ious_list = []
        best_aps_list = []
        best_sorted_ap_list = []
        best_boxes_len_list = []
        best_conf_threshold_list = []

        sorted_ious_lists = []
        aps_lists = []
        sorted_ap_lists = []
        boxes_len_lists = []
        conf_threshold_lists = []

        for ap_metrics in self.ap_metrics_list:
            best_sorted_ious, best_aps, best_sorted_ap, best_boxes_len, best_conf_threshold = (
                ap_metrics.get_best_sorted_ap()
            )
            best_sorted_ious_list.append(best_sorted_ious)
            best_aps_list.append(best_aps)
            best_sorted_ap_list.append(best_sorted_ap)
            best_boxes_len_list.append(best_boxes_len)
            best_conf_threshold_list.append(best_conf_threshold)

            ap_metrics.compute_sorted_ap()
            sorted_ious_lists.append(ap_metrics.sorted_ious_list)
            aps_lists.append(ap_metrics.aps_list)
            sorted_ap_lists.append(ap_metrics.sorted_ap_list)
            boxes_len_lists.append(ap_metrics.boxes_len_list)
            conf_threshold_lists.append(ap_metrics.conf_threshold_list)

        results = {
            "best_sorted_ious_list": best_sorted_ious_list,
            "best_aps_list": best_aps_list,
            "best_sorted_ap_list": best_sorted_ap_list,
            "best_conf_threshold_list": best_conf_threshold_list,
            "sorted_ious_lists": sorted_ious_lists,
            "aps_lists": aps_lists,
            "sorted_ap_lists": sorted_ap_lists,
            "conf_threshold_lists": conf_threshold_lists,
            "legend_list": self.legend_list,
        }
        with open(save_path, "w") as fp:
            json.dump(results, fp, sort_keys=True)
