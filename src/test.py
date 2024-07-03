import time

from utils import RICH_PRINTING


wait_time = 3.0
iters = 100
iters_total_wait_time = 1.0
iters_wait_time = iters_total_wait_time / iters


@RICH_PRINTING.running_message()
def f0():
    for i in RICH_PRINTING.pbar(range(5), leave=True):
        for j in RICH_PRINTING.pbar(range(iters // 5), leave=False):
            time.sleep(iters_wait_time)
    time.sleep(wait_time)


@RICH_PRINTING.running_message()
def f1():
    time.sleep(wait_time)
    RICH_PRINTING.print("I'm in f1.")
    time.sleep(wait_time)
    for i in RICH_PRINTING.pbar(range(iters), leave=False):
        time.sleep(iters_wait_time)
        if i == iters // 2:
            break
    time.sleep(wait_time)
    RICH_PRINTING.print("I'm still in f1.")
    time.sleep(wait_time)


@RICH_PRINTING.running_message()
def f2():
    time.sleep(wait_time)
    f0()
    RICH_PRINTING.print("Look at me!")


@RICH_PRINTING.running_message()
def f3():
    time.sleep(wait_time)
    f1()
    time.sleep(wait_time)
    f2()


@RICH_PRINTING.running_message()
def f4():
    time.sleep(wait_time)
    RICH_PRINTING.print("Here I am!")
    f2()
    time.sleep(wait_time)
    f3()
    time.sleep(wait_time)
    f0()


@RICH_PRINTING.running_message()
def f5():
    RICH_PRINTING.print("Thingy")


@RICH_PRINTING.running_message()
def f6():
    time.sleep(wait_time)
    RICH_PRINTING.print("A")
    time.sleep(wait_time)
    f5()
    time.sleep(wait_time)
    RICH_PRINTING.print("B")
    time.sleep(wait_time)
    for i in RICH_PRINTING.pbar(range(5), leave=True):
        for j in RICH_PRINTING.pbar(range(iters // 5), leave=False):
            time.sleep(iters_wait_time)
    time.sleep(wait_time)
    RICH_PRINTING.print("C")
    time.sleep(wait_time)
    f5()
    time.sleep(wait_time)
    RICH_PRINTING.print("D")


@RICH_PRINTING.running_message()
def main():
    time.sleep(wait_time)
    f6()


# from rich.traceback import install


# install(show_locals=True)

if __name__ == "__main__":
    main()
    time.sleep(wait_time)
    # main()
    # time.sleep(wait_time)
    # main()
    # time.sleep(wait_time)
    # main()
    # time.sleep(wait_time)
    # main()
    RICH_PRINTING.close()
    pass

# from typing import Callable, Iterable, TypeVar

# T = TypeVar("T")

# class PBarIterator:
    # def __init__(
        # self, iterable: Iterable[T], callback_iter: Callable[[], None], callback_end_iter: Callable[[bool], None]
    # ):
        # self.iterator = iter(iterable)
        # self.index = 0
        # self.callback_iter = callback_iter
        # self.callback_end_iter = callback_end_iter
        # self._done = False

    # def __iter__(self):
        # return self

    # def __next__(self):
        # try:
            # if self.index > 0:
                # self.callback_iter()

            # item = next(self.iterator)
            # self.index += 1
            # return item
        # except StopIteration:
            # if not self._done:
                # self.callback_end_iter(False)
                # self._done = True
            # raise StopIteration

    # def __del__(self):
        # if not self._done:
            # self.callback_end_iter(True)
            # self._done = True


# # Example usage
# def iter_callback():
    # print("Item processed.")

# def end_callback(interrupted):
    # if interrupted:
        # print("Iteration was interrupted.")
    # else:
        # print("Iteration completed.")

# iterable = [1, 2, 3, 4, 5]
# pbar_iter = PBarIterator(iterable, iter_callback, end_callback)

# for item in pbar_iter:
    # print(item)