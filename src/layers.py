import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models.utils.loss import DETRLoss
from ultralytics.nn.modules.block import Bottleneck, C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from ultralytics.utils.tal import make_anchors

from box_cls import Box
from cbam import CBAM
from utils import Folders, download_file


sys.path.append(Folders.GOLD_YOLO.value)

from yolov6.models.yolo import build_network, make_divisible
from yolov6.utils.config import Config


def get_scale_constants(scale: str) -> Tuple[float, float, float]:
    """Returns `(depth, width, ratio)` corresponding to the scale.

    Args:
        scale (str): scale of the model (among `['n', 's', 'm', 'l', 'x']`)

    Outputs:
        depth (float)
        width (float)
        ratio (float)
    """
    if scale == "n":
        return (0.33, 0.25, 2.0)
    if scale == "s":
        return (0.33, 0.50, 2.0)
    if scale == "m":
        return (0.67, 0.75, 1.5)
    if scale == "l":
        return (1.00, 1.00, 1.0)
    if scale == "x":
        return (1.00, 1.25, 1.0)
    raise Exception(f"'{scale}' is not an available scale")


class Conv0(Conv):
    """Convolution layer with the right BatchNorm2d parameters."""

    default_act = nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)


class Bottleneck0(Bottleneck):
    """Bottleneck layer with the right BatchNorm2d parameters."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv0(c1, c_, k[0], 1)
        self.cv2 = Conv0(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2


class C2f0(C2f):
    """C2f layer with the right BatchNorm2d parameters."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv0(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv0((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck0(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )


class DetectCustom(Detect):
    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = ()):
        super().__init__(nc, ch)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x

    def preds_from_output(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y


class YOLOv8Backbone(nn.Module):
    """Backbone of the YOLOv8 model (except SPPF)"""

    INPUT_SIZE = 640

    def __init__(self, scale: str = "n", c_input: int = 3) -> None:
        """Initializes the backbone of the YOLOv8 model (except SPPF) with the chosen scale.

        Args:
            scale (str, optional): scale of the model (among ["n", "s", "m", "l", "x"]). Defaults to "n".
            c_input (int, optional): number of channels of the input Tensor.
        """
        super().__init__()
        self.scale = scale
        self.default_state_path = f"../models/yolov8{self.scale}_backbone.pt"
        self.c_input = c_input

        # Create the model
        depth, width, ratio = get_scale_constants(scale)
        self.chs = [
            c_input,
            round(64 * width),
            round(128 * width),
            round(256 * width),
            round(512 * width),
            round(512 * width * ratio),
        ]
        ns = [round(3 * depth), round(6 * depth)]
        self.conv1 = Conv0(self.chs[0], self.chs[1], 3, 2)  # P1
        self.conv2 = Conv0(self.chs[1], self.chs[2], 3, 2)
        self.c2f2 = C2f0(self.chs[2], self.chs[2], ns[0], True)  # P2
        self.conv3 = Conv0(self.chs[2], self.chs[3], 3, 2, 1)
        self.c2f3 = C2f0(self.chs[3], self.chs[3], ns[1], True)  # P3
        self.conv4 = Conv0(self.chs[3], self.chs[4], 3, 2, 1)
        self.c2f4 = C2f0(self.chs[4], self.chs[4], ns[1], True)  # P4
        self.conv5 = Conv0(self.chs[4], self.chs[5], 3, 2, 1)
        self.c2f5 = C2f0(self.chs[5], self.chs[5], ns[0], True)  # P5

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[-2:]
        if shape[0] > shape[1]:
            resized_shape = (
                self.INPUT_SIZE,
                round(self.INPUT_SIZE * shape[1] / shape[0]),
            )
        else:
            resized_shape = (
                round(self.INPUT_SIZE * shape[0] / shape[1]),
                self.INPUT_SIZE,
            )
        resize = transforms.Resize(resized_shape, antialias=True)
        return resize(x)

    def forward(
        self, x: torch.Tensor, output_indices: List[int] | None = None
    ) -> List[torch.Tensor]:
        x = self.preprocess(x)
        outputs = []
        outputs.append(self.conv1(x))
        outputs.append(self.c2f2(self.conv2(outputs[-1])))
        outputs.append(self.c2f3(self.conv3(outputs[-1])))
        outputs.append(self.c2f4(self.conv4(outputs[-1])))
        outputs.append(self.c2f5(self.conv5(outputs[-1])))
        if output_indices is None:
            return [outputs[-1]]
        else:
            return list(map(lambda i: outputs[i - 1], output_indices))

    def load_from_real_state(self, state_path: str | None = None):
        if state_path is None:
            state_path = self.default_state_path
        state_dict = torch.load(state_path)
        self.load_state_dict(state_dict)

    def load_from_yolo_original(
        self,
        first_conv: str,
        state_path: str | None = None,
        download_if_missing: bool = True,
    ):
        first_conv_values = ["exact", "average"]
        if first_conv not in first_conv_values:
            raise Exception(f"first_conv must be chosen among {first_conv_values}.")
        if first_conv == "exact" and self.c_input != 3:
            raise Exception("first_conv == 'exact' is only possible if self.c_input == 3.")
        if state_path is None:
            state_path = f"../models/yolov8{self.scale}.pt"
        if isinstance(state_path, str):
            if not os.path.exists(state_path):
                if download_if_missing:
                    url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8{self.scale}.pt"
                    download_file(url, state_path)
                else:
                    raise Exception(f"There is no file at {os.path.abspath(state_path)}.")
        state_dict = torch.load(state_path)

        # Dictionary of equivalence between layers
        layer_name_dict = {"model.0": "conv1"}
        for i in range(4):
            layer_name_dict[f"model.{2*i+1}"] = f"conv{i+2}"
            layer_name_dict[f"model.{2*i+2}"] = f"c2f{i+2}"

        # Iterate over the items in the state dictionary
        for name, param in state_dict["model"].named_parameters():
            # Check if the layer exists in the model
            layer = name.split(".")
            layer = ".".join(layer[:2])
            if layer in layer_name_dict.keys():
                real_name = name.replace(layer, layer_name_dict[layer], 1)
                weights = param.data
                # Load the weights
                if name == "model.0.conv.weight" and first_conv == "average":
                    weights_averaged = torch.mean(weights, dim=1, keepdim=True)
                    weights = weights_averaged.repeat(1, self.c_input, 1, 1)
                self.state_dict()[real_name].data.copy_(weights)

    def save_real_state(self, state_path: str | None = None):
        if state_path is None:
            state_path = self.default_state_path
        state_dict = self.state_dict()
        torch.save(state_dict, state_path)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Divides and rearranges the channels in the tensor.

    Implementation from here https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py

    Args:
        x (torch.Tensor): input tensor
        groups (int): number of groups to divide channels in

    Returns:
        torch.Tensor: output with rearranged channels
    """
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

    return x


class AMF(nn.Module):
    """Attention Multi-level Fusion layer."""

    def __init__(self, c1_left: int, c1_right: int, r: int) -> None:
        """Initializes an AMF layer.

        Args:
            c1_left (int): channels of the left branch.
            c1_right (int): channels of the right branch.
            r (int): reduction ratio.
        """
        super().__init__()
        self.cbam_left = CBAM(c1_left, r=r)
        self.cbam_right = CBAM(c1_right, r=r)
        self.shuffle_groups = 2
        c2 = c1_left + c1_right
        self.conv = Conv0(c2, c2, k=1, s=1, p=0)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        first_cat = torch.cat((self.cbam_left(x_left), self.cbam_right(x_right)), dim=1)
        # return torch.cat((first_cat, self.conv(self.channel_shuffle(first_cat))), dim=1)
        long_branch = self.conv(channel_shuffle(first_cat, self.shuffle_groups))
        return torch.cat((first_cat, long_branch), dim=1)


class AMFNet(nn.Module):
    """Attention Multi-level Fusion Network."""

    def __init__(
        self, c_input_left: int, c_input_right: int, scale: str = "n", r: int = 16
    ) -> None:
        """Initializes and AMFNet

        Args:
            c_input_left (int): input channels of the left branch.
            c_input_right (int): input channels of the right branch.
            scale (str, optional): scale of the YOLOv8 backbones (among ["n", "s", "m", "l", "x"]). Defaults to "n".
            r (int, optional): reduction ratio of AMF layers. Defaults to 16.
        """
        super().__init__()
        self.yolo_backbone_left = YOLOv8Backbone(scale, c_input_left)  # RGB images
        self.yolo_backbone_right = YOLOv8Backbone(scale, c_input_right)  # LiDAR data
        self.amfs = nn.ModuleList(
            [
                AMF(self.yolo_backbone_left.chs[i], self.yolo_backbone_right.chs[i], r)
                for i in range(2, 6)
            ]
        )

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> List[torch.Tensor]:
        layers = [2, 3, 4, 5]
        left_outputs: Tuple[torch.Tensor, ...] = self.yolo_backbone_left(x_left, layers)
        right_outputs: Tuple[torch.Tensor, ...] = self.yolo_backbone_right(x_right, layers)
        return list(
            map(
                lambda i: self.amfs[i - 2](left_outputs[i - 2], right_outputs[i - 2]),
                layers,
            )
        )


class GD(nn.Module):

    def __init__(self, config_file: str) -> None:
        """Initializes a Gather and Distribute (GD) model using

        Args:
            config_file (str): path of a file containing the configuration of the model.
        """
        super().__init__()
        config = Config.fromfile(config_file)
        if not hasattr(config, "training_mode"):
            setattr(config, "training_mode", "repvgg")
        num_layers = config.model.head.num_layers
        _, self.model, _ = build_network(
            config,
            channels=3,
            num_classes=10,
            num_layers=num_layers,
            fuse_ab=False,
            distill_ns=False,
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return list(self.model.forward(tuple(x)))


class AMF_GD_YOLOv8(nn.Module):
    def __init__(
        self,
        c_input_left: int,
        c_input_right: int,
        class_names: Dict[int, str],
        device: torch.device,
        name: str,
        scale: str = "n",
        r: int = 16,
        loss_weights: Dict[str, float] = {"box": 20, "cls": 1, "dfl": 10},
        gd_config_file: str | None = None,
    ) -> None:
        super().__init__()

        # Store
        self.class_names = class_names
        self.class_indices = {value: key for key, value in class_names.items()}
        self.name = name

        # AMFNet structure
        self.amfnet = AMFNet(c_input_left, c_input_right, scale, r).to(device)

        # Gather and Distribute structure
        if gd_config_file is None:
            gd_config_file = os.path.join(Folders.GD_CONFIGS.value, f"gold_yolo-{scale}.py")
        self.gd = GD(gd_config_file).to(device)

        # Detection structure
        depth, width, ratio = get_scale_constants(scale)
        self.gd_config = Config.fromfile(gd_config_file)
        channels_list = [
            make_divisible(i * width, 8)
            for i in (
                self.gd_config.model.backbone.out_channels + self.gd_config.model.neck.out_channels
            )
        ]
        out_channels = (channels_list[6], channels_list[8], channels_list[10])
        self.detect = DetectCustom(len(class_names), out_channels).to(device)
        self.detect.stride = torch.tensor([8, 16, 32])

        # Whole model
        self.model = nn.ModuleList([self.amfnet, self.gd, self.detect])

        # Loss function
        required_loss_weights_keys = ["box", "cls", "dfl"]
        if set(loss_weights.keys()) != set(required_loss_weights_keys):
            raise ValueError(f"The keys of `loss_weights` should be {required_loss_weights_keys}.")

        class Args:
            def __init__(self) -> None:
                self.box = loss_weights["box"]
                self.cls = loss_weights["cls"]
                self.dfl = loss_weights["dfl"]

        self.args = Args()
        self.criterion = TrainingLoss(self)

    def _open_image(self, image_path: str) -> torch.Tensor:
        to_tensor_transform = transforms.ToTensor()

        image = Image.open(image_path)
        image_tensor = to_tensor_transform(image)
        return image_tensor.unsqueeze(0)

    def _pre_process(
        self,
        x_left: torch.Tensor | str,
        x_right: torch.Tensor | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x_left, str):
            x_left = self._open_image(x_left)
        if isinstance(x_right, str):
            x_right = self._open_image(x_right)
        return x_left, x_right

    def forward(
        self,
        x_left: torch.Tensor | str,
        x_right: torch.Tensor | str,
    ) -> List[torch.Tensor]:
        x_left, x_right = self._pre_process(x_left=x_left, x_right=x_right)
        xs = self.amfnet(x_left, x_right)
        xs = self.gd(xs)
        output = self.detect.forward(xs)
        return output

    @torch.no_grad()
    def preds_from_output(self, output: List[torch.Tensor]):
        return self.detect.preds_from_output(output)

    @torch.no_grad()
    def predict_from_preds(
        self,
        preds: torch.Tensor,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> Tuple[List[List[Box]], List[List[float]], List[List[int]]]:
        # Extract the results
        boxes_list, scores_list, classes_list = extract_bboxes(
            preds,
            iou_thres=iou_threshold,
            conf_thres=conf_threshold,
        )
        return boxes_list, scores_list, classes_list

    @torch.no_grad()
    def predict(
        self,
        x_left: torch.Tensor | str,
        x_right: torch.Tensor | str,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ) -> Tuple[List[List[Box]], List[List[float]], List[List[int]]]:
        output = self.forward(x_left, x_right)
        preds = self.preds_from_output(output)
        return self.predict_from_preds(
            preds, iou_threshold=iou_threshold, conf_threshold=conf_threshold
        )

    def compute_loss(
        self,
        output: List[torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = {"cls": gt_classes, "bboxes": gt_bboxes, "batch_idx": gt_indices}
        return self.criterion(output, batch)

    @staticmethod
    def _get_model_name(index: int, epochs: int, postfix: str | None = None) -> str:
        if postfix is None:
            model_name = f"trained_model_{epochs}ep_{index}"
        else:
            model_name = f"trained_model_{postfix}_{epochs}ep_{index}"
        return model_name

    @staticmethod
    def get_model_path_from_name(model_name: str | None = None) -> str:
        model_path = os.path.join(Folders.MODELS_AMF_GD_YOLOV8.value, f"{model_name}.pt")
        return model_path

    @staticmethod
    def get_last_model_name_and_path(epochs: int, postfix: str | None = None) -> Tuple[str, str]:
        index = 0
        model_name = AMF_GD_YOLOv8._get_model_name(index, epochs, postfix)
        model_path = AMF_GD_YOLOv8.get_model_path_from_name(model_name)
        if not os.path.exists(model_path):
            raise Exception("No such model exists.")
        while os.path.exists(model_path):
            index += 1
            model_name = AMF_GD_YOLOv8._get_model_name(index, epochs, postfix)
            model_path = AMF_GD_YOLOv8.get_model_path_from_name(model_name)

        model_name = AMF_GD_YOLOv8._get_model_name(index - 1, epochs, postfix)
        model_path = AMF_GD_YOLOv8.get_model_path_from_name(model_name)
        return model_name, model_path

    @staticmethod
    def get_new_model_name_and_path(epochs: int, postfix: str | None = None) -> Tuple[str, str]:
        index = 0
        model_name = AMF_GD_YOLOv8._get_model_name(index, epochs, postfix)
        model_path = AMF_GD_YOLOv8.get_model_path_from_name(model_name)
        while os.path.exists(model_path):
            index += 1
            model_name = AMF_GD_YOLOv8._get_model_name(index, epochs, postfix)
            model_path = AMF_GD_YOLOv8.get_model_path_from_name(model_name)

        return model_name, model_path


class TrainingLoss(v8DetectionLoss):
    """Loss used for the training of the model. Based on v8DetectionLoss from ultralytics,
    with a few small tweaks regarding data format.
    """

    def preprocess(self, targets, batch_size, scale_tensor):
        """Pre-processes the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor)) # Original line
        return out

    def __call__(
        self, preds: List[torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), torch.tensor(1))

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        total_loss = loss.sum() * batch_size
        loss_items = loss.detach()  # loss(box, cls, dfl)

        loss_dict = {
            "Box Loss": batch_size * loss_items[0],
            "Class Loss": batch_size * loss_items[1],
            "Dual Focal Loss": batch_size * loss_items[2],
            "fg_mask.sum()": fg_mask.sum(),
            "target_scores_sum": target_scores_sum.clone().detach(),
        }
        return total_loss, loss_dict


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
        containing the predicted boxes, classes, and masks. The tensor should be in the format
        output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
        Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
        Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
        classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        max_det (int): The maximum number of boxes to keep after NMS.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
        shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
        (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if in_place:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    else:
        prediction = torch.cat(
            (xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1
        )  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output


def extract_bboxes(
    preds: torch.Tensor,
    iou_thres: float,
    conf_thres: float,
) -> Tuple[List[List[Box]], List[List[float]], List[List[int]]]:
    batch_size = preds.shape[0]
    preds_nms = non_max_suppression(preds, conf_thres, iou_thres)

    boxes_list = [list(map(Box.from_list, preds_nms[i][:, :4])) for i in range(batch_size)]
    scores_list = [list(map(float, preds_nms[i][:, 4])) for i in range(batch_size)]
    classes_list = [list(map(int, preds_nms[i][:, 5])) for i in range(batch_size)]
    return boxes_list, scores_list, classes_list


def swap_image_b_r(image: Image.Image) -> Image.Image:
    r, g, b = image.split()
    temp = r
    r = b
    b = temp
    return Image.merge("RGB", (r, g, b))
