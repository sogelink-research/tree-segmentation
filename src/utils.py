from __future__ import annotations

import bisect
import json
import os
import random
import shutil
import string
import time
import uuid
import warnings
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import tifffile
import urllib3
from osgeo import gdal
from PIL import Image
from requests import get
from rich.console import Console, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.style import StyleType
from rich.text import Text
from rich.tree import Tree
from tqdm import tqdm

from box_cls import Box
from dataset_constants import DatasetConst


gdal.UseExceptions()
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


def _absolute_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


class Folders(Enum):
    DATA = _absolute_path("../data/")

    TEMP = _absolute_path("../data/temp/")

    ANNOTS = _absolute_path("../data/annotations/")
    FULL_ANNOTS = _absolute_path("../data/annotations/full/")
    CROPPED_ANNOTS = _absolute_path("../data/annotations/cropped/")

    IMAGES = _absolute_path("../data/images/")

    RGB_IMAGES = _absolute_path("../data/images/rgb/")
    FULL_RGB_IMAGES = _absolute_path("../data/images/rgb/full/")
    CROPPED_RGB_IMAGES = _absolute_path("../data/images/rgb/cropped/")

    CIR_IMAGES = _absolute_path("../data/images/cir/")
    FULL_CIR_IMAGES = _absolute_path("../data/images/cir/full/")
    CROPPED_CIR_IMAGES = _absolute_path("../data/images/cir/cropped/")

    LIDAR = _absolute_path("../data/lidar/")
    GEOTILES_LIDAR = _absolute_path("../data/lidar/geotiles/")
    GEOTILES_NO_OVERLAP_LIDAR = _absolute_path("../data/lidar/geotiles_no_overlap/")
    UNFILTERED_FULL_LIDAR = _absolute_path("../data/lidar/unfiltered/full/")
    UNFILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/unfiltered/cropped/")
    FILTERED_FULL_LIDAR = _absolute_path("../data/lidar/filtered/full/")
    FILTERED_CROPPED_LIDAR = _absolute_path("../data/lidar/filtered/cropped/")

    OTHERS_DIR = _absolute_path("../data/others")
    OUTPUT_DIR = _absolute_path("../data/others/model_output")

    MODELS_AMF_GD_YOLOV8 = _absolute_path("../models/amf_gd_yolov8")
    GD_CONFIGS = _absolute_path("../models/gd_configs")

    CHM = _absolute_path("../data/chm/")

    GOLD_YOLO = _absolute_path("../src/Efficient-Computing/Detection/Gold-YOLO")


def create_folder(folder_path: str) -> str:
    """Creates the folder if it doesn't exist, otherwise does nothing.

    Args:
        folder_path (str): path to the folder to create.

    Returns:
        str: the absolute path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.abspath(folder_path)


def create_all_folders() -> None:
    """Creates all the data folders if they don't already exist."""
    for folder in Folders:
        create_folder(folder.value)


def remove_folder(folder_path: str) -> bool:
    """Removes the folder if it exists, as well as its content.

    Args:
        folder_path (str): path to the folder to remove if it exists.

    Returns:
        bool: whether a folder was removed.
    """
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            return True
        except OSError as e:
            RICH_PRINTING.print("Error: %s - %s." % (e.filename, e.strerror))
            return False
    return False


def remove_all_files_but(folder_path: str, files_to_keep: List[str]) -> None:
    """Removes all the files in a folder except some of them.

    Args:
        folder_path (str): path to the folder where files should be removed.
        files_to_keep (List[str]): names (without folder path) of the files to keep.
    """
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file not in files_to_keep:
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                except OSError as e:
                    RICH_PRINTING.print("Error: %s - %s." % (e.filename, e.strerror))


def get_files_in_folders(folders_paths: List[str]) -> List[str]:
    """Returns a list containing the paths to all the files in all the folders given as input.

    Args:
        folders_paths (List[str]): list of paths to folders.

    Returns:
        List[str]: list of paths to files contained in the folders.
    """
    all_files = []
    for folder_path in folders_paths:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
    return all_files


def download_file(url: str, save_path: str, no_ssl: bool = False, verbose: bool = True) -> None:
    """Downloads a file from a URL and saves it at the given path.
    If a file already exists at this path, nothing is downloaded.

    Args:
        url (str): URL to download from.
        save_path (str): path to save the downloaded file.
        no_ssl (bool, optional): if True, the SSL certificate check is skipped. Defaults to False.
        verbose (bool, optional): whether to print messages about the behavior of the function. Defaults to True.
    """
    if os.path.exists(save_path):
        if verbose:
            RICH_PRINTING.print(
                f"Download skipped: there is already a file at '{os.path.abspath(save_path)}'."
            )
        return
    # Send a GET request to the URL
    if verbose:
        RICH_PRINTING.print(f"Downloading {url}... ")
    if no_ssl:
        response = get(url, verify=False)
    else:
        response = get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        with open(save_path, "wb") as f:
            f.write(response.content)
        if verbose:
            RICH_PRINTING.print(f"Done.\nSaved at '{os.path.abspath(save_path)}'.")
    else:
        if verbose:
            RICH_PRINTING.print(
                f"Failed to download file from '{url}'. Status code: {response.status_code}"
            )


class RunningMessage:
    def __init__(self) -> None:
        self.messages: List[str] = []
        self.levels: List[int] = []
        self.kinds: List[str] = []
        self.current_level = 0
        self.last_calls_line_per_level = []
        self.pbars: Dict[int, tqdm] = {}

    def _store_new_line(self, message: str, kind: str, pbar: Optional[tqdm] = None) -> None:
        available_kinds = ["start", "end", "pbar", "print"]
        if kind not in available_kinds:
            raise Exception(f"kind should be one of {available_kinds}, not {kind}.")

        current_line = len(self.messages)
        self.messages.append(message)
        self.levels.append(self.current_level)
        self.kinds.append(kind)

        if kind == "start":
            self.last_calls_line_per_level.append(current_line)
        elif kind == "end":
            self.last_calls_line_per_level.pop(-1)
        elif kind == "pbar":
            if pbar is None:
                raise Exception("The tqdm progress bar should be stored as well.")
            self.pbars[current_line] = pbar

    def _remove_line(self, line: int) -> None:
        self.messages.pop(line)
        self.levels.pop(line)
        kind = self.kinds.pop(line)
        if kind == "pbar":
            self.pbars.pop(line)

    def _get_message_indentation(self, line: int) -> str:
        level = self.levels[line]
        vertical_indent_level = bisect.bisect_left(self.last_calls_line_per_level, line)
        empty_indent_length = 0 if level == 0 else max(vertical_indent_level, 1)
        indent_spaces = (empty_indent_length) * "   " + (level - empty_indent_length) * "│  "
        return indent_spaces

    def _get_arrow_first_char(self, line: int) -> str:
        level = self.levels[line]
        kind = self.kinds[line]
        last_line = len(self.messages) - 1
        # Level 0
        if level == 0:
            return "─"
        else:  # level > 0
            if kind == "start":
                # If the function is still running
                if (
                    len(self.last_calls_line_per_level) > level
                    and self.last_calls_line_per_level[level] == line
                ):
                    return "└"
                # If the function has ended
                else:
                    return "├"
            else:  # kind in ["end", "print", "pbar"]:
                next_line = line + 1
                # while True:
                #     next_line += 1
                #     if next_line >= len(self.kinds) or self.kinds[next_line] in ["start", "end"]:
                #         break

                # If nothing after
                if line == last_line:
                    return "└"
                # If the message after is on a lower level
                elif self.levels[next_line] < level:
                    return "└"
                # If the message after is on the same level
                else:
                    return "├"

    def _get_full_message(self, line: int) -> str:
        message = self.messages[line]
        level = self.levels[line]
        kind = self.kinds[line]
        last_line = len(self.messages) - 1

        indentation_spaces = self._get_message_indentation(line)

        # First character
        first_char = self._get_arrow_first_char(line)

        # First connection
        if kind == "end":
            first_connection = ""
        else:  # kind in ["start", "pbar", "print"]
            first_connection = "──"

        ### Second character
        next_line = line + 1
        # Message of higher level below
        if line < last_line and self.levels[next_line] > level:
            second_char = "┬"
        else:
            second_char = "─"

        # Second connection
        if kind == "end":
            second_connection = ""
        else:  # kind in ["start", "pbar", "print"]
            second_connection = "─"

        # Arrow char
        if kind == "start":
            arrow_char = "►"
        elif kind == "end":
            arrow_char = "◄"
        elif kind == "print":
            arrow_char = "☛"
        elif kind == "pbar":
            arrow_char = "↻"

        full_message = (
            indentation_spaces
            + first_char
            + first_connection
            + second_char
            + second_connection
            + arrow_char
            + " "
            + message
        )
        return full_message

    def _write_lines(self, start: int, end: int):
        total_lines = len(self.messages)
        tqdm.write("\033[F" * (total_lines - start))
        for line in range(start, end):
            if self.kinds[line] == "pbar":
                if line != 4:
                    indentation = self._get_message_indentation(line)
                    arrow_first_char = self._get_arrow_first_char(line)
                    tqdm.write(indentation + arrow_first_char + "\033[E", end="")
            else:
                full_message = self._get_full_message(line)
                tqdm.write(
                    "\033[K" + full_message,
                )
        if (total_lines - end) > 0:
            tqdm.write("\033[E" * (total_lines - end))

        # sys.stdout.flush()

    def log_start_message(self, message: str):
        self._store_new_line(message, kind="start")
        self._write_lines(0, len(self.messages))
        self.current_level += 1

    def log_end_message(self, message: str):
        self._store_new_line(message, kind="end")
        self._write_lines(0, len(self.messages))
        self.current_level -= 1

    def running_message(
        self, start_message: Optional[str] = None, end_message: Optional[str] = None
    ):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Set default messages if not provided
                start_msg = (
                    start_message if start_message is not None else f"Running {func.__name__}..."
                )
                end_msg = end_message if end_message is not None else "Done in {:.4f} seconds."

                # Print the start message
                self.log_start_message(start_msg)

                # Execute the function
                start_time = time.time()
                result = func(*args, **kwargs)
                exec_time = time.time() - start_time

                # Print the end message
                self.log_end_message(end_msg.format(exec_time))

                return result

            return wrapper

        return decorator

    def tqdm(self, *args, desc: str | None = None, leave: bool = False, **kwargs):
        pbar = tqdm(*args, desc=desc, leave=leave, **kwargs)
        current_line = len(self.messages)

        self._store_new_line("", kind="pbar", pbar=pbar)

        indentation = self._get_full_message(current_line)

        def new_set_description(desc: str | None = None, refresh: bool | None = True) -> None:
            pbar.desc = indentation + desc + ": " if desc else indentation
            if refresh:
                pbar.refresh()

        # time.sleep(4)

        pbar.set_description = new_set_description
        pbar.set_description(desc)

        time.sleep(4)

        self._write_lines(0, current_line)

        time.sleep(4)

        tqdm.write("\033[F\033[F")

        if not leave:
            self._remove_line(current_line)
        return pbar

    def print(
        self,
        *values: str,
        sep: str = " ",
    ):
        self._store_new_line(sep.join(values), kind="print")
        self._write_lines(0, len(self.messages))


RUNNING_MESSAGE = RunningMessage()


class PBarList(list):
    T = TypeVar("T")

    def __init__(
        self, sequence: List[T], callback_iter: Callable[[],], callback_end_iter: Callable[[],]
    ) -> None:
        super().__init__(sequence)
        self.length = len(sequence)
        self.progress = 0
        self.callback_iter = callback_iter
        self.callback_end_iter = callback_end_iter

    def __iter__(self):
        return PBarIterator(
            self, callback_iter=self.callback_iter, callback_end_iter=self.callback_end_iter
        )


class PBarIterator(Iterator):
    T = TypeVar("T")

    def __init__(
        self, sequence: PBarList, callback_iter: Callable[[],], callback_end_iter: Callable[[],]
    ):
        self.sequence = sequence
        self.index = 0
        self.callback_iter = callback_iter
        self.callback_end_iter = callback_end_iter

    def __next__(self):
        if self.index > 0:
            self.callback_iter()

        if self.index < len(self.sequence):
            item = self.sequence[self.index]
            self.index += 1
            return item
        else:
            self.callback_end_iter()
            raise StopIteration


class NodeFormat:
    def __init__(
        self,
        label: RenderableType,
        style: StyleType = "tree",
        guide_style: StyleType = "tree.line",
        expanded: bool = True,
        highlight: bool = False,
    ) -> None:
        self.label = label
        self.style = style
        self.guide_style = guide_style
        self.expanded = expanded
        self.highlight = highlight

    def to_tree(self) -> Tree:
        return Tree(
            label=self.label,
            style=self.style,
            guide_style=self.guide_style,
            expanded=self.expanded,
            highlight=self.highlight,
        )

    @staticmethod
    def from_tree(tree: Tree) -> NodeFormat:
        # print(f"{tree.label = }")
        node_format = NodeFormat(
            label=tree.label,
            style=tree.style,
            guide_style=tree.guide_style,
            expanded=tree.expanded,
            highlight=tree.highlight,
        )
        return node_format


def tree_copy_without_children(tree: Tree) -> Tree:
    return NodeFormat.from_tree(tree).to_tree()


class RichPrinting:

    def __init__(self) -> None:
        self._reset_attributes()
        self._reset()

    def _reset_attributes(self) -> None:
        self.trees: Dict[uuid.UUID, Tree] = {}
        self.parents: Dict[uuid.UUID, uuid.UUID] = {}
        self.children: Dict[uuid.UUID, List[uuid.UUID]] = defaultdict(list)
        self.pos_as_child: Dict[uuid.UUID, int] = {}
        self.stack: List[uuid.UUID] = []
        self.stack_order_index: List[int] = []
        self.nodes_in_order: List[uuid.UUID] = []
        self.pbars_progress: Dict[uuid.UUID, Progress] = {}

        self.current_level = 0
        self.leave = True
        self.renderable_up_to_date = False
        self.renderable: RenderableType = Text()

    def _print_all_end(self):
        if hasattr(self, "live"):
            self.render(force_full=True)

    def _reset(self) -> None:
        self._print_all_end()
        if hasattr(self, "live"):
            self.live.stop()

        self.console = Console()
        self.live = Live(
            console=self.console,
            auto_refresh=False,
            # refresh_per_second=4,
            # vertical_overflow="visible",
            # get_renderable=self.get_renderable,
        )

        self._reset_attributes()

    def _get_new_uuid(self) -> uuid.UUID:
        while True:
            new_uuid = uuid.uuid4()
            if new_uuid not in self.trees.keys():
                return new_uuid

    def _store_new_line(self, new_node: Tree, kind: str) -> uuid.UUID:
        available_kinds = ["start", "end", "pbar", "print"]
        if kind not in available_kinds:
            raise Exception(f"kind should be one of {available_kinds}, not {kind}.")

        new_uuid = self._get_new_uuid()

        if self.current_level == 0:
            self._reset()
            self.root_id = new_uuid
            self.nodes_in_order.append(new_uuid)
        else:
            # TODO: improve nested pbars
            if self.nodes_in_order[-1] in self.pbars_progress.keys() and kind == "pbar":
                new_uuid = self.nodes_in_order[-1]
            else:
                parent_uuid = self.stack[self.current_level - 1]
                self.trees[parent_uuid].children.append(new_node)
                self.children[parent_uuid].append(new_uuid)
                self.pos_as_child[new_uuid] = len(self.trees[parent_uuid].children) - 1
                self.parents[new_uuid] = parent_uuid
                self.nodes_in_order.append(new_uuid)

        self.trees[new_uuid] = new_node

        self.renderable_up_to_date = False

        return new_uuid

    def _remove_line(self, id: uuid.UUID) -> None:
        if id in self.stack:
            raise Exception("Cannot remove a line that is still in the stack")

        parent = self.parents.pop(id)
        child_index = self.pos_as_child.pop(id)
        self.trees[parent].children.pop(child_index)

        # Decrease the child index of nodes with the same parent
        for other_id in self.children[parent][child_index + 1 :]:
            self.pos_as_child[other_id] -= 1
        self.children[parent].pop(child_index)

        if id in self.pbars_progress:
            self.pbars_progress.pop(id)

        self.trees.pop(id)
        self.nodes_in_order.remove(id)

    @property
    def _get_terminal_height(self) -> int:
        terminal_size = os.get_terminal_size()
        terminal_height = terminal_size.lines
        return terminal_height

    def _update_renderable(self) -> None:
        if self.renderable_up_to_date:
            return
        self.renderable_up_to_date = True

        terminal_height = self._get_terminal_height
        if terminal_height > len(self.trees):
            self.renderable = self.trees[self.root_id]
            return

        last_uuid = self.nodes_in_order[-1]

        # Count the space of the stack
        stack_height = len(self.stack)
        for id in self.stack[1:]:
            if self.pos_as_child[id] > 0:
                stack_height += 1
        if self.pos_as_child[last_uuid] > 0:
            stack_height += 1

        if stack_height >= terminal_height:
            self.renderable = tree_copy_without_children(self.trees[self.root_id])
            current_height = 1
            last_node = self.renderable
            for id in self.stack[1:]:
                if self.pos_as_child[id] > 0:
                    if terminal_height - current_height >= 4:
                        last_node.add(" ··· ", style="yellow")
                        current_height += 1
                        still_place = True
                    else:
                        still_place = False
                else:
                    if terminal_height - current_height >= 3:
                        still_place = True
                    else:
                        still_place = False

                if still_place:
                    last_node = last_node.add(tree_copy_without_children(self.trees[id]))
                else:
                    last_node = last_node.add(" ··· ", style="yellow")
                    last_node.add(tree_copy_without_children(self.trees[last_uuid]))
                    return
            return

        # Get the last lines to fill the space
        remaining_height = terminal_height - stack_height
        last_common_parent_order_index = (
            bisect.bisect_right(self.stack_order_index, len(self.nodes_in_order) - remaining_height)
            - 1
        )
        if last_common_parent_order_index == -1:
            raise Exception("This should not happen.")
        else:
            last_common_parent_id = self.nodes_in_order[
                self.stack_order_index[last_common_parent_order_index]
            ]

        first_last_line_order_index = len(self.nodes_in_order) - remaining_height
        while (
            first_last_line_order_index < len(self.nodes_in_order) - 1
            and self.parents[self.nodes_in_order[first_last_line_order_index]]
            != last_common_parent_id
        ):
            first_last_line_order_index += 1

        # Build the tree
        tree_equivs: Dict[uuid.UUID, Tree] = {}
        self.renderable = tree_copy_without_children(self.trees[self.root_id])
        tree_equivs[self.root_id] = self.renderable
        for id in self.stack[1:]:
            if self.pos_as_child[id] > 0:
                tree_equivs[self.parents[id]].add(" ··· ", style="yellow")
            tree_equivs[id] = tree_equivs[self.parents[id]].add(
                tree_copy_without_children(self.trees[id])
            )
            if id == last_common_parent_id:
                break

        first_last_line_id = self.nodes_in_order[first_last_line_order_index]
        first_last_line_parent_id = self.parents[first_last_line_id]
        if first_last_line_parent_id not in tree_equivs:
            if self.pos_as_child[first_last_line_parent_id] > 0:
                tree_equivs[self.parents[first_last_line_parent_id]].add(" ··· ", style="yellow")
            tree_equivs[first_last_line_parent_id] = tree_equivs[
                self.parents[first_last_line_parent_id]
            ].add(tree_copy_without_children(self.trees[first_last_line_parent_id]))
        if self.pos_as_child[first_last_line_id] > 0:
            tree_equivs[self.parents[first_last_line_id]].add(" ··· ", style="yellow")
        tree_equivs[first_last_line_id] = tree_equivs[self.parents[first_last_line_id]].add(
            tree_copy_without_children(self.trees[first_last_line_id])
        )
        # except Exception as e:
        #     print(f"{remaining_height = }")
        #     print(f"{len(self.nodes_in_order) - remaining_height = }")
        #     print(f"{self.stack_order_index = }")
        #     print(f"{last_common_parent_order_index = }")
        #     print(f"{first_last_line_order_index = }")
        #     print(f"{len(self.nodes_in_order) = }")
        #     print(f"{first_last_line_order_index = }")
        #     print(f"{tree_equivs[self.root_id].label = }")

        #     for key in tree_equivs:
        #         print(f"{self.trees[key].label = }")
        #         print(f"{tree_equivs[key].label = }")
        #     print(f"{remaining_height = }")
        #     print(f"{len(self.nodes_in_order) - remaining_height = }")
        #     print(f"{self.stack_order_index = }")
        #     print(f"{last_common_parent_order_index = }")
        #     print(f"{len(self.nodes_in_order) = }")
        #     print(f"{first_last_line_order_index = }")
        #     print(f"{self.trees[first_last_line_id].label = }")
        #     print(f"{self.trees[self.parents[first_last_line_id]].label = }")
        #     raise e

        for id in self.nodes_in_order[first_last_line_order_index + 1 :]:
            tree_equivs[id] = tree_equivs[self.parents[id]].add(
                tree_copy_without_children(self.trees[id])
            )

        return

    def get_renderable(self, force_full: bool = False) -> RenderableType:
        if hasattr(self, "root_id"):
            if force_full:
                return self.trees[self.root_id]
            self._update_renderable()
            return self.renderable
        else:
            return Text()

    def close(self):
        if hasattr(self, "live"):
            self._print_all_end()
            self.live.stop()

    def render(self, force_full: bool = False):
        if not self.live.is_started:
            self.live.start()
        self.live.update(self.get_renderable(force_full), refresh=True)

    def _format_message(self, message: str, kind: str) -> NodeFormat:
        label = message

        # Text style
        if kind == "start":
            style = "green"
        elif kind == "end":
            style = "blue"
        elif kind == "print":
            style = "purple"
        else:
            raise Exception(f"Unsupported kind: {kind}")

        node_format = NodeFormat(label=label, style=style)

        return node_format

    def log_start_message(self, message: str):
        node_format = self._format_message(message, kind="start")
        node = node_format.to_tree()
        new_uuid = self._store_new_line(node, kind="start")
        self.stack.append(new_uuid)
        self.stack_order_index.append(len(self.nodes_in_order) - 1)
        self.render()
        self.current_level += 1

    def log_end_message(self, message: str):
        node_format = self._format_message(message, kind="end")
        node = node_format.to_tree()
        self._store_new_line(node, kind="end")
        self.render()
        self.stack.pop(-1)
        self.stack_order_index.pop(-1)
        self.current_level -= 1

    def running_message(
        self, start_message: Optional[str] = None, end_message: Optional[str] = None
    ):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Set default messages if not provided
                start_msg = (
                    start_message if start_message is not None else f"Running {func.__name__}..."
                )
                end_msg = end_message if end_message is not None else "Done in {:.4f} seconds."

                # Print the start message
                self.log_start_message(start_msg)

                # Execute the function
                start_time = time.time()
                result = func(*args, **kwargs)
                exec_time = time.time() - start_time

                # Print the end message
                self.log_end_message(end_msg.format(exec_time))

                return result

            return wrapper

        return decorator

    T = TypeVar("T")

    def pbar(self, iterable: Iterable[T], description: str = "", leave: bool = False) -> List[T]:
        sequence = list(iterable)
        length = len(sequence)

        last_uuid = self.nodes_in_order[-1]
        if last_uuid not in self.pbars_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                SpinnerColumn(spinner_name="arrow"),
                auto_refresh=False,
            )
        else:
            progress = self.pbars_progress[last_uuid]

        task = progress.add_task(description, total=length)
        new_node = Tree(progress.get_renderable())
        new_uuid = self._store_new_line(new_node, kind="pbar")
        self.pbars_progress[new_uuid] = progress

        if leave:

            def callback_end_iter():
                progress.remove_task(task)
                pass

        else:
            if last_uuid == new_uuid:

                def callback_end_iter():
                    progress.remove_task(task)

            else:

                def callback_end_iter():
                    progress.remove_task(task)
                    self._remove_line(new_uuid)

        pbar_sequence = PBarList(
            sequence,
            callback_iter=lambda: self._pbar_update(new_uuid, task),
            callback_end_iter=callback_end_iter,
        )
        return pbar_sequence

    def _pbar_update(self, id: uuid.UUID, taskID: TaskID) -> None:
        progress = self.pbars_progress[id]
        progress.update(taskID, advance=1)

        parent_id = self.parents[id]
        new_node = Tree(progress.get_renderable())
        child_index = self.pos_as_child[id]
        self.trees[parent_id].children[child_index] = new_node
        self.trees[id] = new_node

        self.render()

    def print(
        self,
        *values: object,
        sep: str = " ",
    ):
        str_values = [value.__str__() for value in values]
        message = sep.join(str_values)
        node_format = self._format_message(message, kind="print")
        node = node_format.to_tree()
        self._store_new_line(node, kind="print")

        self.render()


RICH_PRINTING = RichPrinting()


def get_file_base_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def open_json(json_file_path: str) -> Dict[Any, Any]:
    with open(json_file_path, "r") as file:
        return json.load(file)


def get_coordinates_from_full_image_file_name(file_name: str) -> Tuple[int, int]:
    splitted_name = file_name.split("_")
    return (int(splitted_name[-4]), int(splitted_name[-3]))


def get_coordinates_bbox_from_full_image_file_name(file_name: str) -> Box:
    x, y = get_coordinates_from_full_image_file_name(file_name)
    image_size = 1000
    return Box(x_min=x, y_min=y - image_size, x_max=x + image_size, y_max=y)


def get_pixels_bbox_from_full_image_file_name(file_name: str) -> Box:
    image_size = 12500
    return Box(x_min=0, y_min=0, x_max=image_size, y_max=image_size)


class ImageData:
    def __init__(self, image_path: str) -> None:
        self.path = image_path
        self._init_properties()
        self.base_name = get_file_base_name(self.path)
        self.coord_name = f"{round(self.coord_box.x_min)}_{round(self.coord_box.y_max)}"
        # self.coord_box = get_coordinates_bbox_from_full_image_file_name(self.base_name)
        # self.pixel_box = get_pixels_bbox_from_full_image_file_name(self.base_name)

    def _init_properties(self):
        ds = gdal.Open(self.path)

        # Get the geotransform and projection
        gt = ds.GetGeoTransform()

        # Get the extent of the TIF image
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + gt[1] * ds.RasterXSize
        y_min = y_max + gt[5] * ds.RasterYSize

        self.coord_box = Box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        self.width_pixel: int = ds.RasterXSize
        self.height_pixel: int = ds.RasterYSize
        self.pixel_box = Box(x_min=0, y_min=0, x_max=self.width_pixel, y_max=self.height_pixel)

        ds = None


def generate_random_name(length: int = 8):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def create_random_temp_folder(length: int = 8) -> str:
    def get_folder_name(random_name: str) -> str:
        return os.path.join(Folders.TEMP.value, random_name)

    temp_folder_path = get_folder_name(generate_random_name(length))
    while os.path.isdir(temp_folder_path):
        temp_folder_path = get_folder_name(generate_random_name(length))

    create_folder(temp_folder_path)

    return temp_folder_path


def is_tif_file(file_path: str) -> bool:
    return file_path.lower().endswith((".tif", ".tiff"))


def is_npy_file(file_path: str) -> bool:
    return file_path.lower().endswith(".npy")


def write_numpy(image: np.ndarray, save_path: str) -> None:
    mmapped_array = np.lib.format.open_memmap(
        save_path, mode="w+", shape=image.shape, dtype=image.dtype
    )
    mmapped_array[:] = image[:]
    mmapped_array.flush()


def write_image(image: np.ndarray, chm: bool, save_path: str) -> None:
    if is_tif_file(save_path):
        dtype_type = DatasetConst.CHM_DATA_TYPE.value if chm else DatasetConst.RGB_DATA_TYPE.value
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
        tifffile.imwrite(save_path, image)
    elif is_npy_file(save_path):
        dtype_type = np.float32
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
        write_numpy(image, save_path)
    else:
        im = Image.fromarray(image)
        im.save(save_path)


def convert_tif_to_memmappable(image_path, output_path):
    with tifffile.TiffFile(image_path) as tif:
        image = tif.asarray()
        tifffile.imwrite(output_path, image, dtype=image.dtype)


def read_tif(file_path: str, mode: str) -> np.ndarray:
    try:
        image = tifffile.memmap(file_path)
    except ValueError as e:
        if str(e) == "image data are not memory-mappable":
            memmaped_path = file_path.replace(".tif", "_memmap.tif")
            if not os.path.isfile(memmaped_path):
                RICH_PRINTING.print(
                    f"Warning: {e}. Converting the image to a memory-mappable format."
                )
                convert_tif_to_memmappable(file_path, memmaped_path)
            image = tifffile.memmap(memmaped_path, mode=mode)  # type: ignore
        else:
            raise
    return image


def read_numpy(file_path: str, mode: str) -> np.ndarray:
    return np.lib.format.open_memmap(file_path, mode=mode)


def read_image(image_path: str, chm: bool, mode: str) -> np.ndarray:
    if is_tif_file(image_path):
        image = tifffile.imread(image_path)
        # image = read_tif(image_path, mode)
        dtype_type = DatasetConst.CHM_DATA_TYPE.value if chm else DatasetConst.RGB_DATA_TYPE.value
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
    elif is_npy_file(image_path):
        image = read_numpy(image_path, mode)
        dtype_type = np.float32
        if image.dtype != dtype_type:
            image = image.astype(dtype_type)
    else:
        image = np.array(Image.open(image_path))

    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    return image


def get_sup_dtype_type(dtype_type: type):
    handled_dtype_types = [np.floating, np.unsignedinteger, np.signedinteger]
    for handled_dtype_type in handled_dtype_types:
        if np.issubdtype(dtype_type, handled_dtype_type):
            return handled_dtype_type
    raise TypeError(f"Type {dtype_type} is not handled.")


def are_same_numpy_dtype_type_nature(dtype_type1: type, dtype_type2: type):
    if get_sup_dtype_type(dtype_type1) == get_sup_dtype_type(dtype_type2):
        return True
    else:
        return False


def smallest_numpy_dtype_type(dtype_type1: type, dtype_type2: type) -> type:
    if not are_same_numpy_dtype_type_nature(dtype_type1, dtype_type2):
        raise TypeError(f"{dtype_type1} and {dtype_type2} are not the same kind of dtype.")
    if dtype_type1(0).itemsize > dtype_type2(0).itemsize:
        return dtype_type2
    else:
        return dtype_type1


def crop_dtype_type_precision(array: np.ndarray, dtype_type: type):
    smallest_dtype_type = smallest_numpy_dtype_type(array.dtype.type, dtype_type)
    return array.astype(smallest_dtype_type)


def crop_dtype_type_precision_image(image: np.ndarray):
    if get_sup_dtype_type(image.dtype.type) == np.floating:
        return crop_dtype_type_precision(image, np.float32)
    if get_sup_dtype_type(image.dtype.type) == np.unsignedinteger:
        return crop_dtype_type_precision(image, np.uint8)
    if get_sup_dtype_type(image.dtype.type) == np.signedinteger:
        return crop_dtype_type_precision(image, np.int16)
    raise TypeError(f"Type {image.dtype} is not handled.")
