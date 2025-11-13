import tempfile
import os
import rasterio
from rasterio.warp import transform

def get_coords(tiff_path):
    """
    Returns the latitude and longitude of the top-left corner of a TIFF file.
    
    Args:
        tiff_path (str): Path to the TIFF file
        
    Returns:
        tuple: (latitude, longitude) in decimal degrees
    """
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        lon, lat = transform(src.crs, 'EPSG:4326', [bounds.left], [bounds.top])
        return lat[0], lon[0]