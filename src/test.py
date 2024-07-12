import time

from utils import RICH_PRINTING


wait_time = 1.0
iters = 100
iters_total_wait_time = 1.0
iters_wait_time = iters_total_wait_time / iters


@RICH_PRINTING.running_message()
def f0():
    for i in RICH_PRINTING.pbar(range(5), len(range(5)), leave=True):
        for j in RICH_PRINTING.pbar(range(iters // 5), len(range(iters // 5)), leave=False):
            time.sleep(iters_wait_time)
    time.sleep(wait_time)


@RICH_PRINTING.running_message()
def f1():
    time.sleep(wait_time)
    RICH_PRINTING.print("I'm in f1.")
    time.sleep(wait_time)
    for i in RICH_PRINTING.pbar(range(iters), len(range(iters)), leave=False):
        time.sleep(iters_wait_time)
        if i == iters // 2:
            break
    time.sleep(wait_time)
    RICH_PRINTING.print("I'm still in f1.")
    time.sleep(wait_time)


@RICH_PRINTING.running_message()
def f2():
    time.sleep(wait_time)
    for idx, batch_size in RICH_PRINTING.pbar(
        iter(enumerate(range(5))),
        len(range(5)),
        leave=True,
        description="Simulate training with different epochs",
    ):
        RICH_PRINTING.print(idx, batch_size)
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
def f5(leave1: bool, leave2: bool, total1: int = 5, total2: int = 20):
    RICH_PRINTING.print("Thingy")
    time.sleep(wait_time)
    for i in RICH_PRINTING.pbar(range(total1), total1, leave=leave1):
        for j in RICH_PRINTING.pbar(range(total2), total2, leave=leave2):
            time.sleep(iters_wait_time)


@RICH_PRINTING.running_message()
def f6():
    time.sleep(wait_time)
    RICH_PRINTING.print("A")
    time.sleep(wait_time)
    f5(True, True)
    time.sleep(wait_time)
    RICH_PRINTING.print("B")
    time.sleep(wait_time)
    f5(True, False)
    time.sleep(wait_time)
    RICH_PRINTING.print("C")
    time.sleep(wait_time)
    f5(False, True)
    time.sleep(wait_time)
    RICH_PRINTING.print("D")
    time.sleep(wait_time)
    f5(False, False)
    time.sleep(wait_time)
    RICH_PRINTING.print("E")


@RICH_PRINTING.running_message()
def f7():
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        def __init__(self, num_samples: int, num_features: int):
            self.num_samples = num_samples
            self.num_features = num_features
            self.data = torch.randn(num_samples, num_features)  # Random tensor with given shape

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a dummy dataset with 100 samples, each with 10 features
    dummy_dataset = DummyDataset(num_samples=1000, num_features=10)

    # Create a DataLoader for the dummy dataset
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=True)

    time.sleep(wait_time)
    for i in RICH_PRINTING.pbar(range(5), 5, leave=True, description="Epoch"):
        stream = RICH_PRINTING.pbar(
            dummy_dataloader, len(dummy_dataloader), leave=False, description="Training"
        )
        for data in stream:
            time.sleep(0.001)


@RICH_PRINTING.running_message()
def main():
    time.sleep(wait_time)
    f7()


# from rich.traceback import install


# install(show_locals=True)

if __name__ == "__main__":
    for _ in RICH_PRINTING.pbar(range(3), 3):
        main()
        time.sleep(wait_time)
    pass
