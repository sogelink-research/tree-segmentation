from __future__ import annotations

import atexit
import bisect
import os
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from rich.console import Console, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.style import StyleType
from rich.text import Text
from rich.tree import Tree


class PBarIterable:
    def __init__(
        self,
        iterable: Iterable,
        length: int | None,
        callback_iter: Callable[[],],
        callback_end_iter: Callable[[bool],],
    ) -> None:
        self.iterable = iterable
        self.length = length
        self.callback_iter = callback_iter
        self.callback_end_iter = callback_end_iter
        self._done = False

    def __iter__(self):
        iterable = self.iterable
        callback_iter = self.callback_iter

        iter_index = 0

        try:
            for obj in iterable:
                yield obj

                iter_index += 1

                callback_iter()
        except Exception as e:
            self.close(before_end=True)
            raise e
        finally:
            self.close(before_end=(iter_index == self.length))

    def close(self, before_end: bool):
        if not self._done:
            self.callback_end_iter(before_end)
            self._done = True

    def __del__(self):
        self.close(before_end=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(before_end=True)
        return False  # Do not suppress exceptions


class PBarIterator:

    def __init__(
        self,
        iterable: Iterable,
        callback_iter: Callable[[], None],
        callback_end_iter: Callable[[bool], None],
    ):
        self.iterable = iterable
        self.iterator = iter(self.iterable)
        self.index = 0
        self.callback_iter = callback_iter
        self.callback_end_iter = callback_end_iter
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.index > 0:
                self.callback_iter()

            item = next(self.iterator)
            self.index += 1
            return item

        except StopIteration:
            if not self._done:
                self.callback_end_iter(False)
                self._done = True
            raise StopIteration

    def __del__(self):
        if hasattr(self, "_done") and not self._done:
            self.callback_end_iter(True)
            self._done = True


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
        self.heights: Dict[uuid.UUID, int] = {}
        self.nodes_in_order: List[uuid.UUID] = []
        self.pbars_progress: Dict[uuid.UUID, Progress] = {}

        self.current_level = 0
        self.leave = True
        self.renderable_up_to_date = False
        self.renderable: RenderableType = Text()
        self.last_pbar_update = -np.inf

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
            self.heights[new_uuid] = 1
            self.nodes_in_order.append(new_uuid)
        else:
            if self.nodes_in_order[-1] in self.pbars_progress.keys() and kind == "pbar":
                new_uuid = self.nodes_in_order[-1]
                self.heights[new_uuid] += 1
            else:
                parent_uuid = self.stack[self.current_level - 1]
                self.trees[parent_uuid].children.append(new_node)
                self.children[parent_uuid].append(new_uuid)
                self.pos_as_child[new_uuid] = len(self.trees[parent_uuid].children) - 1
                self.parents[new_uuid] = parent_uuid
                self.heights[new_uuid] = 1
                self.nodes_in_order.append(new_uuid)

        self.trees[new_uuid] = new_node

        self.renderable_up_to_date = False

        return new_uuid

    def _remove_line(self, node_id: uuid.UUID) -> None:
        if node_id in self.stack:
            raise Exception("Cannot remove a line that is still in the stack")

        if node_id != self.root_id:
            parent = self.parents.pop(node_id)
            child_index = self.pos_as_child.pop(node_id)
            self.trees[parent].children.pop(child_index)

            # Decrease the child index of nodes with the same parent
            for other_id in self.children[parent][child_index + 1 :]:
                self.pos_as_child[other_id] -= 1
            self.children[parent].pop(child_index)

        if node_id in self.pbars_progress:
            self.pbars_progress.pop(node_id)

        self.heights.pop(node_id)
        self.trees.pop(node_id)
        self.nodes_in_order.remove(node_id)

    @property
    def _terminal_height(self) -> int:
        terminal_size = os.get_terminal_size()
        terminal_height = terminal_size.lines
        return terminal_height

    @property
    def _current_height(self) -> int:
        return sum(self.heights.values())

    def _update_renderable(self, force_update: bool = False) -> None:
        if not force_update and self.renderable_up_to_date:
            return
        self.renderable_up_to_date = True

        terminal_height = self._terminal_height
        if terminal_height > self._current_height:
            self.renderable = self.trees[self.root_id]
            return

        last_uuid = self.nodes_in_order[-1]

        # Count the space of the stack
        stack_height = 0
        for node_id in self.stack[1:]:
            if self.pos_as_child[node_id] > 0:
                stack_height += 1
            stack_height += self.heights[node_id]
        if self.pos_as_child[last_uuid] > 0:
            stack_height += self.heights[last_uuid]

        if stack_height >= terminal_height:
            self.renderable = tree_copy_without_children(self.trees[self.root_id])
            current_height = 1
            last_node = self.renderable
            for id_idx, node_id in enumerate(self.stack[1:]):
                if self.pos_as_child[node_id] > 0:
                    next_height = self.heights[self.stack[id_idx + 1]]
                    if terminal_height - current_height >= next_height + 2:
                        last_node.add(" ··· ", style="yellow")
                        current_height += 1
                        still_place = True
                    else:
                        still_place = False
                else:
                    if terminal_height - current_height >= next_height + 1:
                        still_place = True
                    else:
                        still_place = False

                if still_place:
                    last_node = last_node.add(tree_copy_without_children(self.trees[node_id]))
                else:
                    last_node = last_node.add(" ··· ", style="yellow")
                    last_node.add(tree_copy_without_children(self.trees[last_uuid]))
                    break
            return

        # Get the last lines to fill the space
        remaining_height = terminal_height - stack_height
        first_last_line_order_index = len(self.nodes_in_order) - 1
        total_height = 0
        while (
            total_height + self.heights[self.nodes_in_order[first_last_line_order_index]]
            < remaining_height
        ):
            total_height += self.heights[self.nodes_in_order[first_last_line_order_index]]
            first_last_line_order_index -= 1

        last_common_parent_order_index = (
            bisect.bisect_right(self.stack_order_index, first_last_line_order_index) - 1
        )
        if last_common_parent_order_index == -1:
            raise Exception("This should not happen.")
        else:
            last_common_parent_id = self.nodes_in_order[
                self.stack_order_index[last_common_parent_order_index]
            ]

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
        for node_id in self.stack[1:]:
            if node_id == last_common_parent_id:
                break
            if self.pos_as_child[node_id] > 0:
                tree_equivs[self.parents[node_id]].add(" ··· ", style="yellow")
            tree_equivs[node_id] = tree_equivs[self.parents[node_id]].add(
                tree_copy_without_children(self.trees[node_id])
            )

        first_last_line_id = self.nodes_in_order[first_last_line_order_index]
        first_last_line_parent_id = self.parents[first_last_line_id]
        if first_last_line_parent_id not in tree_equivs:
            if self.pos_as_child[first_last_line_parent_id] > 0:
                tree_equivs[self.parents[first_last_line_parent_id]].add(" ··· ", style="yellow")
            tree_equivs[first_last_line_parent_id] = tree_equivs[
                self.parents[first_last_line_parent_id]
            ].add(tree_copy_without_children(self.trees[first_last_line_parent_id]))

        if first_last_line_id not in tree_equivs:
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

        for node_id in self.nodes_in_order[first_last_line_order_index + 1 :]:
            tree_equivs[node_id] = tree_equivs[self.parents[node_id]].add(
                tree_copy_without_children(self.trees[node_id])
            )

        return

    def get_renderable(
        self, force_full: bool = False, force_update: bool = False
    ) -> RenderableType:
        if hasattr(self, "root_id"):
            if force_full:
                return self.trees[self.root_id]
            self._update_renderable(force_update)
            return self.renderable
        else:
            return Text()

    def close(self):
        if hasattr(self, "live"):
            self._print_all_end()
            self.live.stop()

    def _time_to_update(self) -> bool:
        return time.time() - self.last_pbar_update >= 0.1

    def render(self, force_full: bool = False, force_update: bool = False) -> None:
        if not self.live.is_started:
            self.live.start()
        self.live.update(self.get_renderable(force_full, force_update), refresh=True)

    def _format_message(self, message: str, kind: str) -> NodeFormat:
        label = message

        # Text style
        if kind == "start":
            style = "bold green"
        elif kind == "end":
            style = "bold blue"
        elif kind == "print":
            style = "bold purple"
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

    def pbar(
        self,
        iterable: Iterable,
        length: int,
        description: str = "",
        leave: bool = False,
    ) -> PBarIterable:
        # Improve by looking instead at the progress bars still running to find out if we're inside one
        if len(self.nodes_in_order) > 0:
            last_uuid = self.nodes_in_order[-1]
        else:
            last_uuid = self._get_new_uuid()
        if last_uuid not in self.pbars_progress:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                "•",
                BarColumn(),
                "•",
                "{task.completed:{task.fields[max_digits]}}/{task.total:{task.fields[max_digits]}}",
                "•",
                TimeRemainingColumn(compact=False, elapsed_when_finished=True),
                "•",
                SpinnerColumn(spinner_name="arrow"),
            )
        else:
            progress = self.pbars_progress[last_uuid]

        max_digits = len(str(length))
        taskID = progress.add_task(description, total=length, max_digits=max_digits)
        new_node = Tree(progress.get_renderable())
        new_uuid = self._store_new_line(new_node, kind="pbar")
        self.pbars_progress[new_uuid] = progress
        self._pbar_update_max_digits(new_uuid)

        # If the previous line was a pbar
        if last_uuid == new_uuid:
            if leave:

                def callback_end_iter(by_break: bool):
                    if by_break:
                        self._pbar_break(new_uuid, taskID)
                    self.current_level -= 1

            else:

                def callback_end_iter(by_break: bool):
                    progress.remove_task(taskID)
                    self.heights[new_uuid] -= 1
                    self._pbar_update_max_digits(new_uuid)
                    self.current_level -= 1

        # If the previous line was something else
        else:
            self.stack.append(new_uuid)
            self.stack_order_index.append(len(self.nodes_in_order) - 1)
            if leave:

                def callback_end_iter(by_break: bool):
                    if by_break:
                        self._pbar_break(new_uuid, taskID)
                    self.stack.pop(-1)
                    self.stack_order_index.pop(-1)
                    self.current_level -= 1

            else:

                def callback_end_iter(by_break: bool):
                    progress.remove_task(taskID)
                    self._remove_line(new_uuid)
                    self.stack.pop(-1)
                    self.stack_order_index.pop(-1)
                    self.current_level -= 1

        pbar_iterable = PBarIterable(
            iterable,
            length,
            callback_iter=lambda: self._pbar_update(new_uuid, taskID),
            callback_end_iter=callback_end_iter,
        )
        self.current_level += 1
        return pbar_iterable

    def _pbar_update(self, node_id: uuid.UUID, taskID: TaskID, force_render: bool = False) -> None:
        progress = self.pbars_progress[node_id]
        progress.update(taskID, advance=1)

        self._pbar_node_update(node_id, force_render)

    def _pbar_break(self, node_id: uuid.UUID, taskID: TaskID) -> None:
        progress = self.pbars_progress[node_id]
        current_progress = progress.tasks[taskID].completed
        progress.update(taskID, total=current_progress)

        self._pbar_node_update(node_id, force_render=True)

    def _pbar_node_update(self, node_id: uuid.UUID, force_render: bool = False) -> None:
        if force_render or self._time_to_update():
            progress = self.pbars_progress[node_id]
            new_node = Tree(progress.get_renderable())
            if node_id != self.root_id:
                parent_id = self.parents[node_id]
                child_index = self.pos_as_child[node_id]
                self.trees[parent_id].children[child_index] = new_node
            self.trees[node_id] = new_node
            self.last_pbar_update = time.time()
            self.render(force_update=True)

    def _pbar_update_max_digits(self, node_id: uuid.UUID) -> None:
        progress = self.pbars_progress[node_id]
        max_digits = max(
            [len(str(task.total)) for task in progress.tasks if task.total is not None]
        )

        for task in progress.tasks:
            task.fields["max_digits"] = max_digits

        self._pbar_node_update(node_id)

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
atexit.register(RICH_PRINTING.close)
