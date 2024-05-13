# Download Infrared Chunk

## Idea

The idea is just to be able to download the infrared data without downloading the file for the whole Netherlands which is huge.

## ChatGPT

Great! If you want to perform the task of fetching data from a WMS and creating a GeoTIFF using Python, you can use libraries like `OWSLib` for interacting with OGC web services (like WMS) and `rasterio` or `GDAL` for handling the raster data. Here's a basic outline of how you could set this up in Python:

### Step 1: Install Required Libraries

First, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install owslib rasterio
```

### Step 2: Python Script to Fetch and Save GeoTIFF

Here’s a simple Python script that demonstrates how to request an image from a WMS server and save it as a GeoTIFF file:

```python
from owslib.wms import WebMapService
import rasterio
from rasterio.io import MemoryFile
 
# URL of the WMS server
wms_url = 'http://example.com/wms?'
 
# Initialize WMS connection
wms = WebMapService(wms_url, version='1.1.1')
 
# WMS layer name and CRS (Coordinate Reference System)
layer = 'your_layer_name'
crs = 'EPSG:4326'
 
# Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
bbox = (-123.07, 49.25, -122.72, 49.31)
 
# Image size
size = (1024, 1024)
 
# GetMap request
response = wms.getmap(layers=[layer],
                      srs=crs,
                      bbox=bbox,
                      size=size,
                      format='image/tiff',
                      transparent=True)
 
# Use rasterio to save the fetched image as a GeoTIFF
with MemoryFile(response.read()) as memfile:
    with memfile.open() as dataset:
        profile = dataset.profile
        data = dataset.read()
 
        with rasterio.open('output.tif', 'w', **profile) as dst:
            dst.write(data)
 
print("GeoTIFF saved as 'output.tif'")
```

### Explanation

- **OWSLib** is used to handle the WMS request.
- **rasterio** manages the raster data fetched from the WMS and saves it as a GeoTIFF file. The `MemoryFile` object is a handy way to handle the image data in memory before writing it to a file.

### Step 3: Run the Script

You can run the script in your Python environment. Make sure to replace the WMS URL, layer name, bounding box, and any other parameters specific to your needs.

This script will give you a GeoTIFF file of the area specified by the bounding box. If you need further customization or additional functionality, let me know!
