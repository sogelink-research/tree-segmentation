import numpy as np
from osgeo import gdal

def compute_chm(dtm_file: str, dsm_file: str, output_file: str):
    # Open DTM and DSM files
    dtm_ds = gdal.Open(dtm_file)
    dsm_ds = gdal.Open(dsm_file)

    # Read raster data as numpy arrays
    dtm_array = dtm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    dsm_array = dsm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    
    no_data_indices_dtm_array = np.where(dtm_array == -9999)
    no_data_indices_dsm_array = np.where(dsm_array == -9999)

    # Calculate CHM
    chm_array = dsm_array - dtm_array
    
    print(f"{chm_array.min() = }")
    print(f"{chm_array.max() = }")
    print(f"{chm_array.mean() = }")
    print(f"{chm_array.std() = }")
    
    # Get rid of missing values
    chm_array[no_data_indices_dtm_array] = -9999
    chm_array[no_data_indices_dsm_array] = -9999
    
    print(f"{chm_array.min() = }")
    print(f"{chm_array.max() = }")
    print(f"{chm_array.mean() = }")
    print(f"{chm_array.std() = }")    

    # Get projection and geotransform information from one of the input files
    projection = dtm_ds.GetProjection()
    geotransform = dtm_ds.GetGeoTransform()

    # Create output raster file
    driver = gdal.GetDriverByName("GTiff")
    chm_ds = driver.Create(output_file, dtm_ds.RasterXSize, dtm_ds.RasterYSize, 1, gdal.GDT_Float32)

    # Write CHM array into the output file
    chm_ds.GetRasterBand(1).WriteArray(chm_array)
    chm_ds.GetRasterBand(1).SetNoDataValue(-9999)

    # Set projection and geotransform
    chm_ds.SetProjection(projection)
    chm_ds.SetGeoTransform(geotransform)

    # Close all datasets
    dtm_ds = None
    dsm_ds = None
    chm_ds = None

    print("CHM calculation completed and saved to", output_file)

if __name__ == "__main__":
    print("Hi!")
    dsm_file = "data/25FN1_13_dsm.tif"
    dtm_file = "data/25FN1_13_dtm_doc.tif"
    chm_file = "data/25FN1_13_chm.tif"
    compute_chm(dtm_file, dsm_file, chm_file)