import numpy as np
from osgeo import gdal
import pdal
from os.path import splitext
import json

gdal.UseExceptions()

RESOLUTION = 0.125

def compute_laz_to_las(laz_file_name: str):
    print("Converting LAZ to LAS...", end="", flush=True)
    file_name = splitext(laz_file_name)[0]
    las_file_name = file_name + ".las"
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": laz_file_name
            },
            {
                "type": "writers.las",
                "filename": las_file_name
            }
        ]
    }
    
    # Execute the pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline))
    count = pipeline.execute()

    # Check if the pipeline execution was successful
    if count == 0:
        print(" Conversion failed.")
    else:
        print(" Conversion successful.")
    
    # laz_ds = gdal.Open(laz_file_name)
    # gdal.Translate(las_file_name, laz_ds, format="las")
    # laz_ds = None
    return laz_file_name

def compute_dsm(las_file_name: str):
    print("Computing Surface Model...", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dsm.tif"
    json = f"""[
    "{las_file_name}",
    {{
        "type":"filters.range",
        "limits":"returnnumber[1:1]"
    }},
    {{
        "type": "writers.gdal",
        "filename":"{output_tif_name}",
        "output_type":"idw",
        "gdaldriver":"GTiff",
        "window_size":4,
        "resolution":{RESOLUTION}
    }}
]"""
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    metadata = pipeline.metadata
    print(f"Done: {count} points found.")
    return output_tif_name
    
def compute_dtm(las_file_name: str):
    print("Computing Terrain Model...", end="", flush=True)
    file_name = splitext(las_file_name)[0]
    output_tif_name = f"{file_name}_dtm.tif"
    json = f"""[
    "{las_file_name}",
    {{
        "type":"filters.smrf",
        "window":33,
        "slope":1.0,
        "threshold":0.15,
        "cell":1.0
    }},
    {{
        "type":"filters.range",
        "limits":"Classification[2:2]"
    }},
    {{
        "type":"writers.gdal",
        "filename":"{output_tif_name}",
        "output_type":"min",
        "gdaldriver":"GTiff",
        "window_size":4,
        "resolution":{RESOLUTION}
    }}
]"""
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    metadata = pipeline.metadata
    print(f" Done: {count} points found.")
    return output_tif_name

def compute_chm(laz_file_name: str):
    las_file_name = compute_laz_to_las(laz_file_name)
    
    # Compute DTM and DSM
    dtm_file_name = compute_dtm(las_file_name)
    dsm_file_name = compute_dsm(las_file_name)
    
    print("Computing Canopy Height Model...", end="", flush=True)
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
    chm_ds = driver.Create(output_tif_name, dtm_ds.RasterXSize, dtm_ds.RasterYSize, 1, gdal.GDT_Float32)

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

    print(f" CHM calculation completed and saved to {output_tif_name}")