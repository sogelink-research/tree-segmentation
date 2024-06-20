import time
from random import shuffle
from typing import Callable, List, Tuple

import h5py
import netCDF4 as nc
import numpy as np
import tifffile
import torch

from layers import AMF_GD_YOLOv8


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
    file_paths: List[str], dtypes: List[np.ndarray], shapes: List[Tuple[int, ...]]
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for file_path, dtype, shape in zip(file_paths, dtypes, shapes):
        mmapped_array = np.memmap(file_path, dtype=dtype, mode="c", shape=shape).__array__()
        images.append(mmapped_array)
    return images


def main():
    tif_test_files = ["Test_image_1.tif", "Test_image_2.tif"]
    hdf5_test_file = "Test_image.h5"
    nc_test_file = "Test_image.nc"
    numpy_test_file = "Test_image.npy"
    memmap_test_files = list(map(lambda s: s.replace(".tif", ".mmap"), tif_test_files))

    images_init = list(map(tifffile.imread, tif_test_files))
    tensors_init = list(map(torch.from_numpy, images_init))

    write_hdf5(images_init, hdf5_test_file)
    write_netCDF4(images_init, nc_test_file)
    write_numpy(images_init, numpy_test_file)
    dtypes, shapes = write_memmap(images_init, memmap_test_files)

    iterations = 50
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
    model = AMF_GD_YOLOv8(
        images_init[0].shape[2], images_init[1].shape[2], {0: "test"}, device=device, name="Test"
    )

    def test(
        read_func: Callable,
        images_init: List[np.ndarray],
        tensors_init: List[torch.Tensor],
        model: AMF_GD_YOLOv8,
        device: torch.device,
        input_paths: str | List[str],
        iterations: int = 5,
        **kwargs,
    ):
        images = read_func(input_paths, **kwargs.get("kwargs", {}))
        images_tensors = list(map(torch.from_numpy, images))
        assert all([np.all(images[i] == images_init[i]) for i in range(len(images))])
        assert all([torch.all(images_tensors[i] == tensors_init[i]) for i in range(len(images))])
        total_load_time = 0.0
        total_to_tensor_time = 0.0
        total_output_time = 0.0
        for _ in range(iterations):
            start_time = time.time()
            images = read_func(input_paths, **kwargs.get("kwargs", {}))
            end_time = time.time()
            total_load_time += end_time - start_time
            start_time = time.time()
            images_tensors = list(
                map(
                    lambda arr: torch.from_numpy(arr)
                    .permute((2, 0, 1))
                    .unsqueeze(0)
                    .to(torch.float32)
                    .to(device),
                    images,
                )
            )
            end_time = time.time()
            total_to_tensor_time += end_time - start_time
            start_time = time.time()
            output = model.forward(images_tensors[0], images_tensors[1])
            print(output[0].shape)
            assert output[0].shape == torch.Size((1, 64, 80, 80))
            end_time = time.time()
            total_output_time += end_time - start_time

        print(f"{read_func.__name__}: ")
        print(f"'load': {total_load_time/iterations:.5f} seconds", end=" ; ")
        print(f"'to tensor': {total_to_tensor_time/iterations:.5f} seconds", end=" ; ")
        print(f"'output':    {total_output_time/iterations:.5f} seconds")

    zipped = list(zip(read_func_list, input_paths_list, kwargs_list))
    shuffle(zipped)
    for read_func, input_paths, kwargs in zipped:
        test(
            read_func=read_func,
            images_init=images_init,
            tensors_init=tensors_init,
            model=model,
            device=device,
            input_paths=input_paths,
            iterations=iterations,
            kwargs=kwargs,
        )


if __name__ == "__main__":
    main()
