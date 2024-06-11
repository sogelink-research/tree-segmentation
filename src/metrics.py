import itertools
from typing import List, Tuple, TypeVar

from matplotlib import pyplot as plt
import numpy as np
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
    threshold: float,
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
            if iou > threshold:
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

    aps = [(tp0) / (p + fn0)]
    sorted_ap = sorted_ious[0] * aps[0]
    for k in range(1, tp0):
        aps.append((tp0 - k) / (p + fn0 + k))
        sorted_ap += 0.5 * (sorted_ious[k] - sorted_ious[k - 1]) * (aps[k] + aps[k - 1])
    return sorted_ious, aps, sorted_ap


def get_sorted_ap_plot(
    sorted_ious: List[float],
    aps: List[float],
    sorted_ap: float,
    show: bool = False,
    save_path: str | None = None,
):
    fig = plt.figure(1, figsize=(10, 6))

    x = [0.0]
    y = [aps[0]]
    x.extend(sorted_ious)
    y.extend(aps)
    x.append(1.0)
    y.append(aps[-1])

    plt.plot(x, y)
    plt.grid(alpha=0.5)
    plt.xlabel("IoU")
    plt.ylabel("AP")
    plt.title(f"Sorted AP = {round(sorted_ap, 4)}")
    plt.tight_layout()

    has_legend, _ = plt.gca().get_legend_handles_labels()
    if any(label != "" for label in has_legend):
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return fig


def main():
    # Example usage
    pred_bboxes = [Box(50, 50, 100, 100), Box(30, 30, 50, 50)]
    pred_labels = ["1", "2"]
    gt_bboxes = [Box(48, 48, 105, 105), Box(35, 35, 60, 60)]
    gt_labels = ["1", "2"]
    threshold = 1e-6

    matched_pairs, unmatched_pred, unmatched_gt = hungarian_algorithm(
        pred_bboxes, pred_labels, gt_bboxes, gt_labels, threshold, agnostic=False
    )
    print("Matched pairs:\n", matched_pairs)
    print("Unmatched predictions:\n", unmatched_pred)
    print("Unmatched ground truth:\n", unmatched_gt)

    sorted_ious, aps, sorted_ap = compute_sorted_ap(matched_pairs, unmatched_pred, unmatched_gt)

    get_sorted_ap_plot(sorted_ious, aps, sorted_ap, save_path="Test.png")

    # image = np.zeros((200, 200, 3), dtype=np.uint8)
    # bboxes = [bbox.as_list() for bbox in list(itertools.chain(*[pred_bboxes, gt_bboxes]))]
    # labels = [label for label in list(itertools.chain(*[pred_labels, gt_labels]))]
    # colors_dict = {"1": (255, 0, 0), "2": (0, 255, 0)}
    # scores = None
    # new_image = create_bboxes_image(
    #     image, bboxes=bboxes, labels=labels, colors_dict=colors_dict, scores=scores
    # )
    # output_path = "Test.png"
    # if output_path is not None:
    #     io.imsave(output_path, new_image)


if __name__ == "__main__":
    main()
