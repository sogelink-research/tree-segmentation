import time
from math import floor, sqrt

import numpy as np

from utils import RICH_PRINTING


wait_time = 0.0
iters = 100
iters_total_wait_time = 5
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
def main():
    time.sleep(wait_time)
    f4()
    RICH_PRINTING.close()


# from rich.traceback import install


# install(show_locals=True)

if __name__ == "__main__":
    # main()
    # time.sleep(wait_time)
    # main()
    # # time.sleep(wait_time)
    # # main()
    # # time.sleep(wait_time)
    # # main()
    # # time.sleep(wait_time)
    # # main()
    # RICH_PRINTING.close()
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

        # Define a ModuleList to hold your modules
        self.module_list = nn.ModuleList(
            [
                nn.Linear(10, 20),  # Example Linear layer
                nn.ReLU(),  # Example ReLU activation
                nn.Linear(20, 5),  # Another Linear layer
            ]
        )

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

    def to_device(self, device):
        # Move each module to the specified device
        for module in self.module_list:
            module.to(device)
        return self


# Usage example:
model = MyModule()

# Print the model architecture
print(model)

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Move the model to the GPU
model.to(device)

# Example input tensor
input_tensor = torch.randn(32, 10).to(device)  # Move input tensor to GPU

# Forward pass through the model
output_tensor = model(input_tensor)
print("Output shape:", output_tensor.shape)
