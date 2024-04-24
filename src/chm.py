import json
import time
from os.path import splitext

import numpy as np
import pdal
from osgeo import gdal

gdal.UseExceptions()

RESOLUTION = 0.25


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time} seconds")
        return result

    return wrapper


def compute_laz_to_las(laz_file_name: str):
    """Decompress LAZ files to LAS and save the new file.

    Return the name of the new LAS file.
    """

    print("Converting LAZ to LAS... ", end="", flush=True)
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
    if count == 0:
        print("Conversion failed.")
    else:
        print("Conversion successful.")

    return laz_file_name


def compute_dsm(las_file_name: str):
    print("Computing Surface Model... ", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dsm.tif"

    pipeline_json = [
        las_file_name,
        {"type": "filters.range", "limits": "returnnumber[1:1]"},
        {
            "type": "writers.gdal",
            "filename": output_tif_name,
            "output_type": "idw",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": RESOLUTION,
        },
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    metadata = pipeline.metadata
    print(f"Done: {count} points found.")
    return output_tif_name


def compute_dtm(las_file_name: str):
    print("Computing Terrain Model... ", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dtm.tif"

    pipeline_json = [
        las_file_name,
        {
            "type": "filters.smrf",
            "window": 33,
            "slope": 1.0,
            "threshold": 0.15,
            "cell": 1.0,
        },
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {
            "type": "writers.gdal",
            "filename": output_tif_name,
            "output_type": "min",
            "gdaldriver": "GTiff",
            "window_size": 4,
            "resolution": RESOLUTION,
        },
    ]
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    metadata = pipeline.metadata
    print(f"Done: {count} points found.")
    return output_tif_name


def compute_chm(laz_file_name: str):
    las_file_name = compute_laz_to_las(laz_file_name)

    # Compute DTM and DSM
    dtm_file_name = compute_dtm(las_file_name)
    dsm_file_name = compute_dsm(las_file_name)

    print("Computing Canopy Height Model... ", end="", flush=True)
    # Output file name
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_chm.tif"

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
        output_tif_name, dtm_ds.RasterXSize, dtm_ds.RasterYSize, 1, gdal.GDT_Float32
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

    print(f"CHM calculation completed and saved to {output_tif_name}")


@measure_execution_time
def compute_laz_minus_ground_height(laz_file_name: str):

    print("Subtract ground height to point cloud... ", end="", flush=True)

    # las_file_name = compute_laz_to_las(laz_file_name)

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
    metadata = pipeline.metadata
    print(f"Done: {count} points found.")

    return output_laz_name
