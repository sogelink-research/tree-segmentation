import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath("Efficient-Computing/Detection/Gold-YOLO"))

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cbam import CBAM
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models.utils.loss import DETRLoss
from ultralytics.nn.modules.block import Bottleneck, C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops
from utils import download_file
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
            Bottleneck0(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )


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
        resize = transforms.Resize(resized_shape, antialias=True)  # type: ignore # antialias=True helps preventing a warning
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
            raise Exception(
                "first_conv == 'exact' is only possible if self.c_input == 3."
            )
        if state_path is None:
            state_path = f"../models/yolov8{self.scale}.pt"
        if isinstance(state_path, str):
            if not os.path.exists(state_path):
                if download_if_missing:
                    url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8{self.scale}.pt"
                    download_file(url, state_path)
                else:
                    raise Exception(
                        f"There is no file at {os.path.abspath(state_path)}."
                    )
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

    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> List[torch.Tensor]:
        layers = [2, 3, 4, 5]
        left_outputs: Tuple[torch.Tensor] = self.yolo_backbone_left(x_left, layers)
        right_outputs: Tuple[torch.Tensor] = self.yolo_backbone_right(x_right, layers)
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
        device: torch.device,
        scale: str = "n",
        r: int = 16,
        gd_config_file: str | None = None,
        class_names: Dict[int, str] = {
            0: "Tree",
            1: "Tree_disappeared",
            2: "Tree_replaced",
            3: "Tree_new",
        },
    ) -> None:
        super().__init__()
        self.class_names = class_names
        self.class_indices = {value: key for key, value in class_names.items()}

        # AMFNet structure
        self.amfnet = AMFNet(c_input_left, c_input_right, scale, r).to(device)

        # Gather and Distribute structure
        if gd_config_file is None:
            gd_config_file = f"../models/gd_configs/gold_yolo-{scale}.py"
        self.gd = GD(gd_config_file).to(device)

        # Detection structure
        depth, width, ratio = get_scale_constants(scale)
        self.gd_config = Config.fromfile(gd_config_file)
        channels_list = [
            make_divisible(i * width, 8)
            for i in (
                self.gd_config.model.backbone.out_channels
                + self.gd_config.model.neck.out_channels
            )
        ]
        out_channels = [channels_list[6], channels_list[8], channels_list[10]]
        self.detect = Detect(len(class_names), out_channels).to(device)

    def open_image(self, image_path: str) -> torch.Tensor:
        to_tensor_transform = transforms.ToTensor()

        image = Image.open(image_path)
        image_tensor = to_tensor_transform(image)
        return image_tensor.unsqueeze(0)

    def forward(
        self,
        x_left: torch.Tensor | str,
        x_right: torch.Tensor | str,
    ) -> torch.Tensor:
        # Open x_left if it is a path
        if isinstance(x_left, str):
            x_left = self.open_image(x_left)

        # Open x_right if it is a path
        if isinstance(x_right, str):
            x_right = self.open_image(x_right)

        # Compute the output
        xs = self.amfnet(x_left, x_right)
        xs = self.gd(xs)
        self.detect.training = False
        y = self.detect(xs)
        return y[0]

    def predict(
        self,
        x_left: torch.Tensor | str,
        x_right: torch.Tensor | str,
        image_save_path: str | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Results]:
        # Open x_left if it is a path
        if isinstance(x_left, str):
            origin_image_path = x_left
            x_left = self.open_image(x_left)
        else:
            origin_image_path = None
        input_img_shape = x_left.shape[2:]
        origin_image = np.ascontiguousarray(
            np.array(np.round(255 * x_left[0].permute(1, 2, 0)), dtype=np.uint8)
        )

        # Open x_right if it is a path
        if isinstance(x_right, str):
            x_right = self.open_image(x_right)

        # Compute the output
        xs = self.amfnet(x_left, x_right)
        xs = self.gd(xs)
        self.detect.training = False
        y = self.detect(xs)

        # Create the results
        result = extract_bboxes(
            y,
            input_img_shape=input_img_shape,
            origin_image=origin_image,
            origin_image_path=origin_image_path,
            class_names=self.class_names,
            confidence=0.5,
            iou=0.5,
        )

        if image_save_path is not None:
            img = result.plot()

            # Convert BGR to RGB (OpenCV uses BGR by default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_save_path, img_rgb)

        return (y[0], y[1], result)

    from typing import Union

    def compute_loss(
        self,
        preds: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_indices: torch.Tensor,
        bboxes_format: str,
    ):
        number_classes = len(self.class_names)
        loss_func = DETRLoss(nc=number_classes)

        if len(preds.shape) == 3:
            preds = preds.unsqueeze(0)

        pred_bboxes = preds.permute((0, 1, 3, 2))[:, :, :, :4].contiguous()
        pred_scores = preds.permute((0, 1, 3, 2))[:, :, :, 4:].contiguous()
        if bboxes_format == "xywh":
            gt_bboxes = xywh2xyxy(gt_bboxes)
        elif bboxes_format != "xyxy":
            raise ValueError("bboxes_format must be 'xywh' or 'xyxy'.")
        batch = {
            "cls": gt_classes.to(dtype=torch.int64),
            "bboxes": gt_bboxes,
            "gt_groups": [idx[1] - idx[0] for idx in gt_indices],
        }

        return loss_func.forward(pred_bboxes, pred_scores, batch)


def extract_bboxes(
    preds: List[torch.Tensor | List[torch.Tensor]],
    input_img_shape: Tuple[int] | torch.Size,
    origin_image: npt.NDArray[np.uint8],
    origin_image_path: str | None,
    class_names: Dict[int, str],
    confidence: float,
    iou: float,
) -> Results:
    """Post-processes predictions and returns a list of Results objects."""
    preds_nms = ops.non_max_suppression(preds, confidence, iou)[0]
    preds_nms[:, :4] = ops.scale_boxes(
        input_img_shape, preds_nms[:, :4], origin_image.shape
    )
    return Results(
        origin_image, path=origin_image_path, names=class_names, boxes=preds_nms
    )


def swap_image_b_r(image: Image.Image) -> Image.Image:
    r, g, b = image.split()
    temp = r
    r = b
    b = temp
    return Image.merge("RGB", (r, g, b))


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Small modification from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    w = x[..., 2]  # half-width
    h = x[..., 3]  # half-height
    y[..., 0] = x[..., 0]  # top left x
    y[..., 1] = x[..., 1]  # top left y
    y[..., 2] = x[..., 0] + w  # bottom right x
    y[..., 3] = x[..., 1] + h  # bottom right y
    return y
