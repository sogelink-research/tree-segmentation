import os
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath("Efficient-Computing/Detection/Gold-YOLO"))

import torch
import torch.nn as nn
from cbam import CBAM
from ultralytics.nn.modules.block import DFL, Bottleneck, C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
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

    def forward(
        self, x: torch.Tensor, output_indices: List[int] | None = None
    ) -> List[torch.Tensor]:
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
        self.channel_shuffle = nn.ChannelShuffle(2)
        c2 = c1_left + c1_right
        self.conv = Conv0(c2, c2, k=1, s=1, p=0)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        first_cat = torch.cat((self.cbam_left(x_left), self.cbam_right(x_right)), dim=1)
        return torch.cat((first_cat, self.conv(self.channel_shuffle(first_cat))), dim=1)


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
        self.amfs = [
            AMF(self.yolo_backbone_left.chs[i], self.yolo_backbone_right.chs[i], r)
            for i in range(2, 6)
        ]

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
        scale: str = "n",
        r: int = 16,
        gd_config_file: str | None = None,
        number_classes: int = 4,
    ) -> None:
        super().__init__()

        # AMFNet structure
        self.amfnet = AMFNet(c_input_left, c_input_right, scale, r)

        # Gather and Distribute structure
        if gd_config_file is None:
            gd_config_file = f"../models/gd_configs/gold_yolo-{scale}.py"
        self.gd = GD(gd_config_file)

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
        self.detect = Detect(number_classes, out_channels)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor):
        xs = self.amfnet(x_left, x_right)
        xs = self.gd(xs)
        self.detect.training = False
        return self.detect(xs)
