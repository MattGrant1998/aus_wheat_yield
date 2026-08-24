import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import alphashape
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
import os
from shapely.geometry import Point
from shapely import vectorized 

datadir = '/g/data/w97/mg5624/ABS_project/'

def create_wheatbelt_shapefile():
    # --- Load data ---
    yield_summary_path = datadir + 'ag_data/'
    avg_yield = xr.open_dataset(yield_summary_path + 'avg_yield.nc')
    da = avg_yield['mean_yield'] if 'mean_yield' in avg_yield else list(avg_yield.data_vars.values())[0]

    lon = da['lon'].values if 'lon' in da.coords else da['longitude'].values
    lat = da['lat'].values if 'lat' in da.coords else da['latitude'].values

    # --- Get coordinates of all valid (non-nan) pixels as points ---
    lon2d, lat2d = np.meshgrid(lon, lat)
    valid_mask = ~np.isnan(da.values)

    valid_lons = lon2d[valid_mask]
    valid_lats = lat2d[valid_mask]

    print(f"Total valid pixels: {len(valid_lons)}")

    # --- Split into west / east by longitude ---
    LON_SPLIT = 129.0  # adjust based on inspecting your data's actual gap

    west_pts = np.column_stack([valid_lons[valid_lons < LON_SPLIT], valid_lats[valid_lons < LON_SPLIT]])
    east_pts = np.column_stack([valid_lons[valid_lons >= LON_SPLIT], valid_lats[valid_lons >= LON_SPLIT]])

    print(f"West points: {len(west_pts)} | East points: {len(east_pts)}")

    # --- Build concave (alpha) hulls ---
    # alpha controls tightness: higher alpha = tighter/more concave, lower = closer to convex hull.
    # Start small and increase if the result looks too convex/bloated; too high can fragment it.
    ALPHA = 1.5

    west_hull = alphashape.alphashape(west_pts, ALPHA)
    east_hull = alphashape.alphashape(east_pts, ALPHA)

    # Alphashape can occasionally return a MultiPolygon if alpha is too high (fragments the points).
    # If that happens, take the union to force one continuous polygon, or lower ALPHA and rerun.
    west_hull = unary_union(west_hull)
    east_hull = unary_union(east_hull)

    # --- Light smoothing of the hull boundary ---
    def smooth(geom, eps=0.05):
        return geom.buffer(eps).buffer(-eps)

    west_smooth = smooth(west_hull)
    east_smooth = smooth(east_hull)

    # --- Combine into a single GeoDataFrame, two rows ---
    gdf_combined = gpd.GeoDataFrame(
        {'region': ['Western Wheatbelt', 'Eastern Wheatbelt']},
        geometry=[west_smooth, east_smooth],
        crs='EPSG:4326'
    )

    out_dir = datadir + 'ag_data/wheatbelt_shapefiles/'
    os.makedirs(out_dir, exist_ok=True)
    gdf_combined.to_file(out_dir + 'wheatbelt.shp')
    return gdf_combined


def create_wheatbelt_nc(shp, grid):
    """
    Build a boolean mask DataArray on grid's lat/lon, True where the
    cell center falls inside any polygon in shp.
    """
    lon_name = 'lon' if 'lon' in grid.coords else 'longitude'
    lat_name = 'lat' if 'lat' in grid.coords else 'latitude'

    lon = grid[lon_name].values
    lat = grid[lat_name].values

    lon2d, lat2d = np.meshgrid(lon, lat)

    # dissolve all polygons (e.g. West + East) into one geometry to test against
    combined_geom = shp.unary_union

    # shapely.vectorized.contains is a fast C-level point-in-polygon test
    # over whole numpy arrays at once, no looping needed
    inside = vectorized.contains(combined_geom, lon2d, lat2d)

    mask_da = xr.DataArray(
        inside,
        dims=(lat_name, lon_name),
        coords={lat_name: lat, lon_name: lon}
    )

    return mask_da


def main():
    wheatbelt_shp = create_wheatbelt_shapefile()
    agcd_datadir = '/g/data/zv2/agcd/v1-0-1/precip/total/r005/01day/'
    grid = xr.open_dataset(agcd_datadir + 'agcd_v1-0-1_precip_total_r005_daily_2016.nc').precip.isel(time=-50)
    wheatbelt_mask = create_wheatbelt_nc(wheatbelt_shp, grid)
    wheatbelt_mask.to_netcdf(datadir + 'ag_data/wheatbelt_shapefiles/wheatbelt.nc')
    print("Saved wheatbelt.shp with 2 smooth continuous polygons (West, East)")


if __name__ == "__main__":
    main()
