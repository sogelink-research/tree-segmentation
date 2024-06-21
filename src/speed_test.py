import time
from itertools import combinations, product
from random import shuffle
from typing import Callable, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import torch


# from layers import AMF_GD_YOLOv8


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


def write_tif(images: List[np.ndarray], save_paths: List[str]) -> None:
    for image, save_path in zip(images, save_paths):
        tifffile.imwrite(save_path, image)


def write_hdf5(images: List[np.ndarray], save_path: str) -> None:
    with h5py.File(save_path, "w") as f:
        for i, image in enumerate(images):
            f.create_dataset(f"image{i}", data=image)


def write_netCDF4(images: List[np.ndarray], save_path: str) -> None:
    with nc.Dataset(save_path, "w") as f:
        f.createDimension("height", images[0].shape[0])
        f.createDimension("width", images[0].shape[1])
        height = f.createVariable("height", np.uint16, ("height",))
        width = f.createVariable("width", np.uint16, ("width",))
        height[:] = images[0].shape[0]
        width[:] = images[0].shape[1]

        for i, image in enumerate(images):
            f.createDimension(f"channels_{i}", image.shape[2])
            channels = f.createVariable(f"channels_{i}", np.uint8, (f"channels_{i}",))
            image_var = f.createVariable(
                f"image{i}", image.dtype, ("height", "width", f"channels_{i}")
            )
            channels[:] = image.shape[2]
            image_var[:] = image


def write_numpy(images: List[np.ndarray], save_path: str) -> None:
    images_dict = {f"image{i}": image for i, image in enumerate(images)}
    np.save(save_path, images_dict)


def write_memmap(images: List[np.ndarray], save_paths: List[str]):
    dtypes: List[np.ndarray] = []
    shapes: List[Tuple[int, ...]] = []
    for image, save_path in zip(images, save_paths):
        mmapped_array = np.memmap(save_path, dtype=image.dtype, mode="w+", shape=image.shape)
        mmapped_array[:] = image[:]
        mmapped_array.flush()
        dtypes.append(image.dtype)
        shapes.append(image.shape)
    return dtypes, shapes


def read_tif(file_paths: List[str]) -> List[np.ndarray]:
    images = list(map(tifffile.imread, file_paths))
    return images


def read_hdf5(file_path: str) -> List[np.ndarray]:
    with h5py.File(file_path, "r") as hf:
        images = list(map(np.array, hf.values()))
    return images


def read_netCDF4(file_path: str) -> List[np.ndarray]:
    with nc.Dataset(file_path, "r") as f:
        idx = 0
        images = []
        while f"image{idx}" in f.variables:
            images.append(np.array(f[f"image{idx}"][:]))
            idx += 1
    return images


def read_numpy(file_path: str) -> List[np.ndarray]:
    data = np.load(file_path, allow_pickle=True).item()
    return list(data.values())


def read_memmap(
    file_paths: List[str],
    dtype_types: List[type],
    shapes: Optional[Sequence[Tuple[int, ...] | None]] = None,
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    shapes = [None] * len(file_paths) if shapes is None else shapes
    for file_path, dtype_type, shape in zip(file_paths, dtype_types, shapes):
        mmapped_array = np.memmap(file_path, dtype=dtype_type, mode="c", shape=shape).__array__()
        if shape is None:
            real_shape = (640, 640, mmapped_array.size // (640 * 640))
            mmapped_array = mmapped_array.reshape(real_shape)
        images.append(mmapped_array)
    return images


def main():
    initial_tif_files = ["Test_image_1.tif", "Test_image_2.tif"]

    tif_test_files = ["Test_image_1_reduced.tif", "Test_image_2_reduced.tif"]
    hdf5_test_file = "Test_image_reduced.h5"
    nc_test_file = "Test_image_reduced.nc"
    numpy_test_file = "Test_image_reduced.npy"
    memmap_test_files = list(map(lambda s: s.replace(".tif", ".mmap"), tif_test_files))

    images_init = list(
        map(crop_dtype_type_precision_image, map(tifffile.imread, initial_tif_files))
    )
    tensors_init = list(map(torch.from_numpy, images_init))

    write_tif(images_init, tif_test_files)
    write_hdf5(images_init, hdf5_test_file)
    write_netCDF4(images_init, nc_test_file)
    write_numpy(images_init, numpy_test_file)
    dtypes, shapes = write_memmap(images_init, memmap_test_files)

    iterations = 100
    read_func_list = [read_tif, read_hdf5, read_netCDF4, read_numpy, read_memmap]
    input_paths_list = [
        tif_test_files,
        hdf5_test_file,
        nc_test_file,
        numpy_test_file,
        memmap_test_files,
    ]
    kwargs_list = [{}, {}, {}, {}, {"dtypes": dtypes, "shapes": shapes}]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AMF_GD_YOLOv8(
    #     images_init[0].shape[2], images_init[1].shape[2], {0: "test"}, device=device, name="Test"
    # )

    def test(
        read_func: Callable,
        images_init: List[np.ndarray],
        tensors_init: List[torch.Tensor],
        # model: AMF_GD_YOLOv8,
        device: torch.device,
        input_paths: str | List[str],
        iterations: int = 5,
        **kwargs,
    ):
        images = read_func(input_paths, **kwargs.get("kwargs", {}))
        images_tensors = list(map(torch.from_numpy, images))
        assert all([np.all(images[i] == images_init[i]) for i in range(len(images))])
        assert all([torch.all(images_tensors[i] == tensors_init[i]) for i in range(len(images))])
        reshaped_tensors_init = list(
            map(
                lambda arr: arr.permute((2, 0, 1))
                .unsqueeze(0)
                .to(torch.float32)
                .to(device, non_blocking=True),
                images_tensors,
            )
        )
        total_load_times = []
        total_to_tensor_times = []
        total_output_times = []
        total_load_process_times = []
        total_to_tensor_process_times = []
        total_output_process_times = []
        entire_start_process_time = time.process_time_ns()
        for _ in range(iterations):
            start_process_time = time.process_time_ns()
            start_time = time.time_ns()

            images = read_func(input_paths, **kwargs.get("kwargs", {}))

            end_process_time = time.process_time_ns()
            end_time = time.time_ns()
            total_load_process_times.append((end_process_time - start_process_time) / 1e9)
            total_load_times.append((end_time - start_time) / 1e9)
            start_process_time = time.process_time_ns()
            start_time = time.time_ns()

            images_tensors = list(
                map(
                    lambda arr: torch.from_numpy(arr)
                    .permute((2, 0, 1))
                    .unsqueeze(0)
                    .to(torch.float32)
                    .to(device, non_blocking=True),
                    images,
                )
            )

            end_process_time = time.process_time_ns()
            end_time = time.time_ns()
            total_to_tensor_process_times.append((end_process_time - start_process_time) / 1e9)
            total_to_tensor_times.append((end_time - start_time) / 1e9)
            start_process_time = time.process_time_ns()
            start_time = time.time_ns()

            # output = model.forward(images_tensors[0], images_tensors[1])
            # assert output[0].shape == torch.Size((1, 64 + len(model.class_indices), 80, 80))
            assert all(
                [
                    torch.all(images_tensors[i] == reshaped_tensors_init[i])
                    for i in range(len(images))
                ]
            )

            end_process_time = time.process_time_ns()
            end_time = time.time_ns()
            total_output_process_times.append((end_process_time - start_process_time) / 1e9)
            total_output_times.append((end_time - start_time) / 1e9)

        entire_end_process_time = time.process_time_ns()
        entire_process_time = (entire_end_process_time - entire_start_process_time) / 1e9

        print(f"{read_func.__name__}: ")
        print("Process time:   ")
        print(
            f"'load':      mean={np.mean(total_load_process_times):.5f} seconds, std={np.std(total_load_process_times):.5f}"
        )
        # print(
        #     f"'to tensor': mean={np.mean(total_to_tensor_process_times):.5f} seconds, std={np.std(total_to_tensor_process_times):.5f}"
        # )
        # print(
        #     f"'output':    mean={np.mean(total_output_process_times):.5f} seconds, std={np.std(total_output_process_times):.5f}"
        # )
        print(f"Entire time per iteration: {entire_process_time:.5f}")
        print("Execution time: ")
        print(
            f"'load':      mean={np.mean(total_load_times):.5f} seconds, std={np.std(total_load_times):.5f}"
        )
        # print(
        #     f"'to tensor': mean={np.mean(total_to_tensor_times):.5f} seconds, std={np.std(total_to_tensor_times):.5f}"
        # )
        # print(
        #     f"'output':    mean={np.mean(total_output_times):.5f} seconds, std={np.std(total_output_times):.5f}"
        # )

        return (
            [
                t1 + t2 + t3
                for (t1, t2, t3) in zip(
                    total_load_process_times,
                    total_to_tensor_process_times,
                    total_output_process_times,
                )
            ],
        )

    zipped = list(zip(read_func_list, input_paths_list, kwargs_list))
    shuffle(zipped)
    total_load_process_times_list = []
    function_names = []
    function_names_once = []
    mean_values = []
    std_values = []
    for read_func, input_paths, kwargs in zipped:
        (total_load_process_times,) = test(
            read_func=read_func,
            images_init=images_init,
            tensors_init=tensors_init,
            # model=model,
            device=device,
            input_paths=input_paths,
            iterations=iterations,
            kwargs=kwargs,
        )
        mean_values.append(np.mean(total_load_process_times))
        std_values.append(np.std(total_load_process_times))
        print(f"{np.min(total_load_process_times)}")
        total_load_process_times_list.extend(total_load_process_times)
        function_names.extend([read_func.__name__] * iterations)
        function_names_once.append(read_func.__name__)

    data1 = {
        "Execution time": total_load_process_times_list,
        "Reading method": function_names,
    }
    df1 = pd.DataFrame(data1)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.violinplot(
        x="Reading method", y="Execution time", data=df1, ax=ax1, order=function_names_once
    )

    # Annotate violin plot with mean and std
    for i, (mean, std) in enumerate(zip(mean_values, std_values)):
        ax1.text(
            i,
            mean + 0.01,
            f"Mean: {mean:.5f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.5),
        )
        ax1.text(
            i,
            mean - 0.01,
            f"Std: {std:.5f}",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.5),
        )
    plt.title("Reading time depending on the format")
    plt.show()


if __name__ == "__main__":
    main()

    # types = [
    #     np.float16,
    #     np.float32,
    #     np.float64,
    #     np.uint8,
    #     np.uint16,
    #     np.uint32,
    #     np.int8,
    #     np.int16,
    #     np.int32,
    #     np.bool_,
    # ]
    # for dtype1, dtype2 in combinations(types, r=2):
    #     try:
    #         array1 = dtype1(22 / 7 * 1e6)
    #         array2 = dtype2(22 / 7 * 1e6)
    #         result = smallest_numpy_dtype_type(array1.dtype.type, array2.dtype.type)
    #         print(f"Smallest type between {array1.dtype.type} and {array2.dtype.type}: {result}")
    #     except TypeError as e:
    #         # print(e)
    #         continue
    #     array3 = crop_dtype_type_precision_image(array1)
    #     array4 = crop_dtype_type_precision_image(array2)
    #     print(f"{array1} --> {array3}")
    #     print(f"{array2} --> {array4}")
