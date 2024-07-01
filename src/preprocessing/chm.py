import json
import os
import subprocess
from os.path import splitext
from typing import Dict, List, Sequence, Tuple

import laspy
import numpy as np
import pdal
from osgeo import gdal

from utils import (
    RICH_PRINTING,
    Folders,
    ImageData,
    create_random_temp_folder,
    remove_folder,
)


gdal.UseExceptions()


def compute_laz_to_las(laz_file_name: str, verbose: bool = False):
    """Decompress LAZ files to LAS and save the new file.

    Return the name of the new LAS file.
    """

    if verbose:
        RICH_PRINTING.print("Converting LAZ to LAS... ", end="", flush=True)
    file_name = splitext(laz_file_name)[0]
    las_file_name = file_name + ".las"
    pipeline_json = [
        {"type": "readers.las", "filename": laz_file_name},
        {"type": "writers.las", "filename": las_file_name},
    ]

    # Execute the pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()

    # Check if the pipeline execution was successful
    if verbose:
        if count == 0:
            RICH_PRINTING.print("Conversion failed.")
        else:
            RICH_PRINTING.print("Conversion successful.")

    return las_file_name


def compute_dsm(
    las_file_name: str,
    width: int,
    height: int,
    resolution: float,
    verbose: bool = False,
):
    if verbose:
        RICH_PRINTING.print("Computing Surface Model... ", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dsm.tif"

    with laspy.open(las_file_name, mode="r") as las_file:
        # Get the bounding box information from the header
        origin_x = las_file.header.min[0]
        origin_y = las_file.header.min[1]

    pipeline_json = [
        las_file_name,
        {"type": "filters.range", "limits": "returnnumber[1:1]"},
        {
            "type": "writers.gdal",
            "filename": output_tif_name,
            "output_type": "idw",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "width": width,
            "height": height,
        },
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    if verbose:
        RICH_PRINTING.print(f"Done: {count} points found. Saved at {output_tif_name}")
    return output_tif_name


def compute_dtm(
    las_file_name: str,
    width: int,
    height: int,
    resolution: float,
    verbose: bool = False,
):
    if verbose:
        RICH_PRINTING.print("Computing Terrain Model... ", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dtm.tif"
    output_tif_name_temp = f"{file_name}_dtm_temp.tif"

    with laspy.open(las_file_name, mode="r") as las_file:
        # Get the bounding box information from the header
        origin_x = las_file.header.min[0]
        origin_y = las_file.header.min[1]

    pipeline_json = [
        las_file_name,
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {
            "type": "writers.gdal",
            "filename": output_tif_name_temp,
            "output_type": "min",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "width": width,
            "height": height,
        },
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()

    old_ds = gdal.Open(output_tif_name_temp)

    # Create output raster file
    driver = gdal.GetDriverByName("GTiff")
    new_ds = driver.CreateCopy(output_tif_name, old_ds)

    band = new_ds.GetRasterBand(1)

    gdal.FillNodata(targetBand=band, maskBand=None, maxSearchDist=1000, smoothingIterations=20)

    # new_ds.GetRasterBand(1).WriteArray(band.ReadAsArray())

    # Close all datasets
    old_ds = None
    new_ds = None

    os.remove(output_tif_name_temp)

    if verbose:
        RICH_PRINTING.print(f"Done: {count} points found. Saved at {output_tif_name}.")
    return output_tif_name


def compute_chm(
    laz_file_name: str,
    output_tif_name: str,
    width: int,
    height: int,
    resolution: float,
    verbose: bool = False,
):
    if os.path.exists(output_tif_name):
        RICH_PRINTING.print(f"The file {os.path.abspath(output_tif_name)} already exists.")
        return
    las_file_name = compute_laz_to_las(laz_file_name, verbose)

    # Compute DTM and DSM
    dtm_file_name = compute_dtm(las_file_name, width, height, resolution, verbose)
    dsm_file_name = compute_dsm(las_file_name, width, height, resolution, verbose)

    if verbose:
        RICH_PRINTING.print("Computing Canopy Height Model... ", end="", flush=True)

    # Open DTM and DSM files
    dtm_ds = gdal.Open(dtm_file_name)
    dsm_ds = gdal.Open(dsm_file_name)

    # Read raster data as numpy arrays
    dtm_array = dtm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    dsm_array = dsm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Calculate CHM
    chm_array = dsm_array - dtm_array

    # Get rid of missing values
    no_data_indices_dtm_array = np.where(dtm_array == -9999)
    no_data_indices_dsm_array = np.where(dsm_array == -9999)
    chm_array[no_data_indices_dtm_array] = -9999
    chm_array[no_data_indices_dsm_array] = -9999

    # Create output raster file
    driver = gdal.GetDriverByName("GTiff")
    chm_ds = driver.Create(
        output_tif_name,
        dtm_ds.RasterXSize,
        dtm_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
    )

    # Write CHM array into the output file
    chm_ds.GetRasterBand(1).WriteArray(chm_array)

    # Set value corresponding to no data
    chm_ds.GetRasterBand(1).SetNoDataValue(-9999)

    # Set projection and geotransform
    chm_ds.SetProjection(dtm_ds.GetProjection())
    chm_ds.SetGeoTransform(dtm_ds.GetGeoTransform())

    # Close all datasets
    dtm_ds = None
    dsm_ds = None
    chm_ds = None

    # Remove intermediary files
    os.remove(las_file_name)
    os.remove(dtm_file_name)
    os.remove(dsm_file_name)

    if verbose:
        RICH_PRINTING.print(f"CHM calculation completed and saved to {output_tif_name}.")


@RICH_PRINTING.running_message("Creating point cloud with flat ground...")
def compute_laz_minus_ground_height(laz_file_name: str, verbose: bool = False):
    if verbose:
        RICH_PRINTING.print("Subtract ground height to point cloud... ", end="", flush=True)

    # las_file_name = compute_laz_to_las(laz_file_name, verbose)

    # Create new file name
    file_name = splitext(laz_file_name)[0]
    output_laz_name = f"{file_name}_minus_gh.LAZ"

    pipeline_json = [
        {"type": "readers.las", "filename": laz_file_name},
        {"type": "filters.smrf"},
        {"type": "filters.hag_nn", "count": 3, "allow_extrapolation": True},
        {"type": "filters.ferry", "dimensions": "HeightAboveGround=>Z"},
        {
            "type": "writers.las",
            "filename": output_laz_name,
        },
    ]

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    if verbose:
        RICH_PRINTING.print(f"Done: {count} points found.")

    return output_laz_name


@RICH_PRINTING.running_message("Creating point cloud with flat ground...")
def compute_laz_minus_ground_height_with_dtm(
    laz_file_name: str, output_laz_name: str, dtm_file_name: str, verbose: bool = False
):
    if verbose:
        RICH_PRINTING.print("Subtract ground height to point cloud... ", end="", flush=True)

    pipeline_json = [
        {"type": "readers.las", "filename": laz_file_name},
        {"type": "filters.hag_dem", "raster": dtm_file_name},
        {"type": "filters.ferry", "dimensions": "HeightAboveGround=>Z"},
        {
            "type": "writers.las",
            "filename": output_laz_name,
        },
    ]

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    if verbose:
        RICH_PRINTING.print(f"Done: {count} points found.")

    return output_laz_name


@RICH_PRINTING.running_message("Creating Digital Terrain Model...")
def compute_full_dtm(
    las_file_name: str,
    output_tif_name: str,
    resolution: float,
    verbose: bool = False,
):
    if verbose:
        RICH_PRINTING.print("Computing Terrain Model... ", end="", flush=True)

    # Create temporary folder
    temp_folder = create_random_temp_folder()

    output_tif_name_temp = os.path.join(temp_folder, "dtm.tif")

    try:
        pipeline_json = [
            las_file_name,
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {
                "type": "writers.gdal",
                "filename": output_tif_name_temp,
                "output_type": "min",
                "gdaldriver": "GTiff",
                "window_size": 4,
                "resolution": resolution,
            },
        ]
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        count = pipeline.execute()

        old_ds = gdal.Open(output_tif_name_temp)

        # Create output raster file
        driver = gdal.GetDriverByName("GTiff")
        new_ds = driver.CreateCopy(output_tif_name, old_ds)

        band = new_ds.GetRasterBand(1)

        # Fill small NO_DATA areas
        gdal.FillNodata(targetBand=band, maskBand=None, maxSearchDist=200, smoothingIterations=20)

        # Replace NO_DATA values with 0
        nodata_value = band.GetNoDataValue()
        data = band.ReadAsArray()
        data[data == nodata_value] = 0

        band.WriteArray(data)

        # Close all datasets
        band.FlushCache()
        old_ds.FlushCache()
        new_ds.FlushCache()
        old_ds = None
        new_ds = None
        if verbose:
            RICH_PRINTING.print(f"Done: {count} points found. Saved at {output_tif_name}.")
    except Exception as e:
        raise e
    finally:
        remove_folder(temp_folder)

    return output_tif_name


def get_read_las_to_stdout_pipeline(
    las_file_name: str,
):
    pipeline_json = [
        {"type": "readers.las", "filename": las_file_name},
        {"type": "writers.text", "filename": "STDOUT"},
    ]

    return pipeline_json


def get_slice_dsm_from_stdin_pipeline(
    output_tif_path: str,
    resolution: float,
    z_limits: Tuple[float, float],
) -> List[Dict]:
    z_limit_bottom = "" if np.isneginf(z_limits[0]) else z_limits[0]
    z_limit_top = "" if np.isposinf(z_limits[1]) else z_limits[1]

    pipeline_json = [
        {"type": "readers.text", "filename": "STDIN"},
        {
            "type": "filters.range",
            "limits": f"returnnumber[1:1],Z[{z_limit_bottom}:{z_limit_top}]",
        },
        {
            "type": "writers.gdal",
            "filename": output_tif_path,
            "output_type": "idw",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": resolution,
        },
    ]
    return pipeline_json


def slow_compute_slices_chm(
    laz_file_name: str,
    output_tif_paths: List[str],
    resolution: float,
    z_limits_list: Sequence[Tuple[float, float]],
) -> None:

    # Create temporary folder
    temp_folder = create_random_temp_folder()

    try:
        slices_pipelines_paths = []
        for i, (z_limits, output_path) in enumerate(zip(z_limits_list, output_tif_paths)):
            pipeline_json = get_slice_dsm_from_stdin_pipeline(output_path, resolution, z_limits)
            slice_pipeline_path = os.path.join(temp_folder, f"{i}.json")
            slices_pipelines_paths.append(slice_pipeline_path)
            with open(slice_pipeline_path, "w") as pipeline_file:
                json.dump(pipeline_json, pipeline_file)

        reading_pipeline_json = get_read_las_to_stdout_pipeline(laz_file_name)
        reading_pipeline_path = os.path.join(temp_folder, "reading_pipeline.json")
        with open(reading_pipeline_path, "w") as file:
            json.dump(reading_pipeline_json, file)

        # Create the bash script
        # "#!/bin/bash\n"
        bash_script = ["#!/usr/bin/env bash\n", f"pdal pipeline {reading_pipeline_path} | ", "tee"]
        for slice_pipeline_path in slices_pipelines_paths:
            bash_script.append(f" >(pdal pipeline {slice_pipeline_path})")
        bash_script.append(" >/dev/null")

        shell_run_path = "data/tests/run.sh"
        with open(shell_run_path, "w") as rsh:
            rsh.write("".join(bash_script))

        subprocess.run(["chmod", "+x", shell_run_path])
        RICH_PRINTING.print(subprocess.run([shell_run_path], shell=True))

    except Exception as e:
        raise e
    finally:
        remove_folder(temp_folder)


def compute_slices_chm_from_hag_laz(
    hag_laz_file_name: str,
    output_tif_paths: List[str],
    resolution: float,
    z_limits_list: Sequence[Tuple[float, float]],
    skip_if_file_exists: bool,
):
    if len(output_tif_paths) != len(z_limits_list):
        raise ValueError(
            "The arguments `output_tif_paths` and `z_limits_list` should have the same length."
        )
    for output_tif_path, z_limits in zip(output_tif_paths, z_limits_list):
        compute_slice_chm_from_hag_laz(
            hag_laz_file_name,
            output_tif_path=output_tif_path,
            resolution=resolution,
            z_limits=z_limits,
            skip_if_file_exists=skip_if_file_exists,
        )


def compute_slice_chm_from_hag_laz(
    hag_laz_file_name: str,
    output_tif_path: str,
    resolution: float,
    z_limits: Tuple[float, float],
    skip_if_file_exists: bool,
):
    if skip_if_file_exists and os.path.isfile(output_tif_path):
        return

    z_limit_bottom = "" if np.isneginf(z_limits[0]) else z_limits[0]
    z_limit_top = "" if np.isposinf(z_limits[1]) else z_limits[1]
    pipeline_json = [
        {"type": "readers.las", "filename": hag_laz_file_name},
        {
            "type": "filters.range",
            "limits": f"returnnumber[1:1],Z[{z_limit_bottom}:{z_limit_top}]",
        },
        {
            "type": "writers.gdal",
            "filename": output_tif_path,
            "output_type": "idw",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": resolution,
        },
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()


@RICH_PRINTING.running_message("Creating slices of Canopy Height Model...")
def compute_slices_chm(
    laz_file_name: str,
    output_tif_paths: List[str],
    resolution: float,
    z_limits_list: Sequence[Tuple[float, float]],
    skip_if_file_exists: bool,
):
    if skip_if_file_exists and all([os.path.isfile(path) for path in output_tif_paths]):
        return

    # Create temporary folder
    temp_folder = create_random_temp_folder()

    try:
        full_dtm_path = os.path.join(temp_folder, "dtm.tif")
        hag_path = os.path.join(temp_folder, "hag.las")

        compute_full_dtm(
            laz_file_name,
            output_tif_name=full_dtm_path,
            resolution=resolution,
        )

        compute_laz_minus_ground_height_with_dtm(
            laz_file_name, output_laz_name=hag_path, dtm_file_name=full_dtm_path
        )

        for z_limits, output_path in zip(z_limits_list, output_tif_paths):
            compute_slice_chm_from_hag_laz(
                hag_path,
                output_tif_path=output_path,
                resolution=resolution,
                z_limits=z_limits,
                skip_if_file_exists=skip_if_file_exists,
            )

    except Exception as e:
        raise e
    finally:
        pass
        # remove_folder(temp_folder)


def get_full_chm_slice_path(
    image_data: ImageData, resolution: float, filtered: bool, z_limits: Tuple[float, float]
):
    filtering_str = "filtered" if filtered else "unfiltered"
    low_limit = str(round(z_limits[0], 1)).replace("-", "M").replace(".", "p")
    high_limit = str(round(z_limits[1], 1)).replace("-", "M").replace(".", "p")
    full_chm_slice_path = os.path.join(
        Folders.CHM.value,
        f"{round(resolution*100)}cm",
        filtering_str,
        f"{low_limit}_{high_limit}",
        "full",
        f"{image_data.coord_name}.tif",
    )
    return full_chm_slice_path
