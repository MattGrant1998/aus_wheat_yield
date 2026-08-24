import xarray as xr
import numpy as np


def load_crop_calendar():
    """
    Load the crop calendar data from a NetCDF file.

    Returns:
        xarray.Dataset: The crop calendar dataset.
    """
    path = '/g/data/w97/mg5624/ABS_project/crop_calendar/Wheat.crop.calendar.nc'
    crop_calendar = xr.open_dataset(path)
    return crop_calendar


def restrict_to_aus(crop_calendar):
    """
    Restrict the crop calendar data to Australia.

    Args:
        crop_calendar (xarray.Dataset): The crop calendar dataset.

    Returns:
        xarray.Dataset: The restricted crop calendar dataset for Australia.
    """
    crop_calendar = crop_calendar.rename({'latitude':'lat', 'longitude':'lon'})
    calendar_flipped = crop_calendar.sortby('lat')
    aus_crop_calendar = calendar_flipped.sel(lat=slice(-45, -10), lon=slice(110, 155))
    return aus_crop_calendar


def regrid_to_agcd(da, year):
    agcd_filepath = '/g/data/zv2/agcd/v1-0-1/precip/total/r005/01day/'
    agcd_data = xr.open_dataset(agcd_filepath + f'agcd_v1-0-1_precip_total_r005_daily_{year}.nc')
    
    max_lat = max(da.lat.max(), agcd_data.lat.max())
    min_lat = min(da.lat.min(), agcd_data.lat.min())
    max_lon = max(da.lon.max(), agcd_data.lon.max())
    min_lon = min(da.lon.min(), agcd_data.lon.min())

    da = da.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    agcd_data = agcd_data.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    da = da.assign_coords(time=agcd_data.time)

    da_regridded = da.interp_like(agcd_data, method='nearest')
    return da_regridded


def create_state_mask(crop_calendar):

    state_lookup = {
        (136, 353.5): 1,
        (166.5, 0.5): 2,
        (151.5, 304): 3,
        (136, 350.5): 4,
        (151, 350.5): 5,
    }

    start = crop_calendar['plant'].values
    end = crop_calendar['harvest'].values
    
    state_mask = np.full(start.shape, np.nan)  # default = no match (outside all states)

    for (p_val, h_val), code in state_lookup.items():
        match = np.isclose(start, p_val) & np.isclose(end, h_val)
        state_mask[match] = code

    # 3. Wrap back into a DataArray on the same grid/coords
    state_da = xr.DataArray(
        state_mask,
        dims=crop_calendar['plant'].dims,
        coords=crop_calendar['plant'].coords,
        name='state_mask',
        attrs={
            'long_name': 'Australian state mask',
            'flag_values': [1, 2, 3, 4, 5],
            'flag_meanings': 'NSW VIC QLD SA WA'
        }
    )

    # Infill anomalous NaNs
    nan_points = (
        (state_da.lat >= -37) & (state_da.lat <= -36.5) &
        (state_da.lon >= 148) & (state_da.lon <= 148.5)
    )

    state_da = state_da.where(~nan_points, 2)
    return state_da


def create_state_wheatbelt_maks(state_mask):
    wheatbelt_mask = xr.open_dataarray('/g/data/w97/mg5624/ABS_project/masks/wheatbelt_mask.nc')
    state_wheat_mask = state_mask.where(wheatbelt_mask == 1)
    return state_wheat_mask


def main():
    agcd_crop_calendar = regrid_to_agcd(
        restrict_to_aus(
            load_crop_calendar()
        ), 
        year=2020
    )

    state_mask = create_state_mask(agcd_crop_calendar)
    state_wheat_mask = create_state_wheatbelt_maks(state_mask)
    
    filepath_out = '/g/data/w97/mg5624/ABS_project/masks/'
    state_mask.to_netcdf(filepath_out + 'state_mask.nc')
    state_wheat_mask.to_netcdf(filepath_out + 'state_wheat_mask.nc')


if __name__ == "__main__":
    main()
