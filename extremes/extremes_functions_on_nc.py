import numpy as np
import pandas as pd
import xarray as xr
import os
import dask
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

data_filepath = '/g/data/w97/mg5624/ABS_project/'

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


def load_agcd_data(variable, year, test=False):
    """
    Loads daily agcd data for the specified year.

    Parameters:
        variable (str): The variable to load data for.
        year (int): The year to load agcd data for.

    Returns:
        xarray.DataArray: Daily agcd data.
    """
    agg = {
        'tmax': 'mean',
        'tmin': 'mean',
        'precip': 'total',
    }
    if variable == 'soil_moisture':
        data_path = f'/g/data/w97/mg5624/RF_project/AWRA_SM/daily/root_zone_soil_moisture_{year}.nc'
        variable = 'awra_root_zone_soil_moisture'
    else:
        data_path = f'/g/data/zv2/agcd/v1-0-1/{variable}/{agg[variable]}/r005/01day/agcd_v1-0-1_{variable}_{agg[variable]}_r005_daily_{year}.nc'
    data = xr.open_dataset(data_path)[variable]
    if variable == 'awra_root_zone_soil_moisture':
        data = data.sortby('latitude')
        data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
        data = data.rename('soil_moisture')
        data = regrid_to_agcd(data, year)
       
    if test:
        data = data.sel(lat=slice(-25, -24), lon=slice(144, 145))
    data = data.chunk(lon=100, lat=100)
    return data


def load_multiple_years_of_agcd(var, start_year, end_year, test=False):
    """
    Load all years of a variable between start_year and end_year.

    Parameters:
        var (str): Variable to load.
        start_year (int): Start year of data to load.
        end_year (int): End year of data to load.

    Returns:
        xr.DataArray: Data for the variable between start_year and end_year.
    """
    all_data = []
    for year in range(start_year, end_year + 1):
        data = load_agcd_data(var, year, test=test)
        all_data.append(data)

    all_data_da = xr.concat(all_data, dim='time')#.chunk(lat=100, lon=100, time=100)
    return all_data_da


def rolling_data(data, n_days, agg, center=False):
    """
    Finds the rolling sum of the data over n_days.

    Parameters:
        data (xr.DataArray): Data to find the rolling sum of.
        var (str): name of variable to find rolling sum of.
        n_days (int): Number of days to find the rolling sum over.
        agg (str: 'sum' or 'mean'): Aggregation method to use. Default is 'sum'.

    Returns:
        xr.DataArray: Data with rolling sum.
    """
    if agg == 'sum':
        rolling_data = data.rolling(time=n_days, center=center).sum()
    elif agg == 'mean':
        rolling_data = data.rolling(time=n_days, center=center).mean()
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")
    rolling_data = rolling_data.rename(f'{data.name}_rolling_{n_days}_{agg}')
    return rolling_data


def find_percentile_at_grid(rolling_data, baseline, threshold):
    """
    Finds the percentile of the data for the specified baseline.

    Parameters:
        rolling_data (xr.DataArray): Rolling sum of the data.
        baseline (str or list): The baseline to use for the percentile. Default is 'full_ts'. Else list of years of form [start_year, end_year].
        threshold (float or str): The percentile to find the threshold of. If finding mean, set threshold='mean'

    Returns:
        xr.DataArray: Percentile of the data.
    """
    if baseline == 'full_ts':
        baseline_data = rolling_data
    else:
        start_year, end_year = baseline
        baseline_data = rolling_data.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

    # Add in Feb 29th as a copy of Feb 28th for years that don't have it
    years = np.unique(baseline_data['time.year'].values)
    non_leap_years = [year for year in years if not pd.Timestamp(f'{year}-12-31').is_leap_year]

    # 2. Duplicate Feb 28 as Feb 29 for non-leap years
    feb28 = baseline_data.sel(
        time=((
            baseline_data['time.month'] == 2
            ) & (baseline_data['time.day'] == 28
                 ) & baseline_data['time.year'].isin(non_leap_years)))

    feb28_coords = feb28['time'].values + np.timedelta64(1, 'D')
    feb28.coords['time'] = ('time', feb28_coords)
    baseline_with_feb29 = xr.concat([baseline_data, feb28], dim='time').sortby('time')

    baseline_with_feb29 = baseline_with_feb29.chunk({'time': -1})
    if threshold == 'mean':
        percentiles = baseline_with_feb29.groupby('time.dayofyear').mean(dim='time')
        var_name = percentiles.name
        percentiles = percentiles.rename(f'{var_name}_mean')
    else:
        percentiles = baseline_with_feb29.groupby('time.dayofyear').quantile(threshold, dim='time')

    return percentiles


def find_drought_days(hydrological_data, var_name, n_days, threshold, threshold_type='percentile', baseline='full_ts'):
    """
    Finds the drought days in the data.

    Parameters:
        hydrological_data (xr.DataArray): Precipitation or soil moisture data to find drought days in.
        var_name (str): Name of the variable to find drought days in.
        n_days (int): Number of days to find the rolling sum over.
        threshold (float): The threshold to find the drought days for.
        threshold_type (str): The type of threshold to use. Default is 'percentile'.

    Returns:
        xr.DataArray: Data with drought days.
    """
    # hydrological_data.to_netcdf(f'/srv/ccrc/LandAU/z5459957/data/extremes/daily/{var_name}.nc')
    centered = False
    hyd_rolling = rolling_data(hydrological_data, n_days, 'mean', center=centered)
   
    if threshold_type == 'percentile':
        # hyd_rolling.coords['dayofyear'] = ('time', hyd_rolling['time'].dt.dayofyear)
        percentile_data = find_percentile_at_grid(hyd_rolling, baseline, threshold)
        thresholds = percentile_data.sel(dayofyear=hyd_rolling['time.dayofyear'])
        drought_days = hyd_rolling.where(hyd_rolling < thresholds)
    elif threshold_type == 'absolute':
        drought_days = hyd_rolling.where(hyd_rolling < threshold)
    else:
        raise ValueError("Invalid threshold type. Use 'percentile' or 'absolute'.")
    drought_days = drought_days.rename(f'{n_days}_day_{var_name}_drought')
    drought_days = xr.merge([drought_days, hyd_rolling])
    print(drought_days)
    return drought_days


def find_heatwave_days(tmax_daily, threshold):
    """
    Find heatwave days in from daily maximum temperature data.

    Parameters:
        tmax_daily (xarray.DataArray): Daily maximum temperature data.
        threshold (float): The temperature threshold to define a heatwave day.

    Returns:
        xarray.DataArray: Daily maximum temperature data for heatwave days.
    """
    # heatwave_threshold = tmax_daily > threshold
    # heatwave_threshold_n_minus_1 = heatwave_threshold.shift(time=1).astype(bool)
    # heatwave_threshold_n_minus_2 = heatwave_threshold.shift(time=2).astype(bool)

    # heatwave_days = tmax_daily.where(
    #     heatwave_threshold & heatwave_threshold_n_minus_1 & heatwave_threshold_n_minus_2
    # )
    heatwave_days = tmax_daily.where(tmax_daily > threshold)
    heatwave_days = heatwave_days.rename('heatwave')
    heatwave_days = xr.merge([heatwave_days, tmax_daily])
    return heatwave_days


def find_frost_days(tmin_daily):
    """
    Find frost days from daily minimum temperature data.

    Parameters:
        tmin_daily (xarray.DataArray): Daily minimum temperature data.

    Returns:
        xarray.DataArray: Daily minimum temperature data for frost days.
    """
    frost_days_mask = tmin_daily < 0
    frost_days = tmin_daily.where(frost_days_mask)
    frost_days = frost_days.rename('frost')
    # frost_days = xr.merge([frost_days, tmin_daily])
    return frost_days


def growing_degree_days(tmax, tmin, basetemp=0):
    """
    Calculate growing degree days (GDD) from daily maximum and minimum temperature data.

    Parameters:
        tmax (xarray.DataArray): Daily maximum temperature data.
        tmin (xarray.DataArray): Daily minimum temperature data.
        basetemp (float): Base temperature for GDD calculation. Default is 0.
    Returns:
        xarray.DataArray: Growing degree days data.
    """
    tavg = (tmax + tmin) / 2
    gdd = tavg - basetemp
    gdd = gdd.where(gdd > 0, 0)
    gdd = gdd.rename('growing_degree_days')
    return gdd


def temperature_range(tmax, tmin):
    """
    Calculate diurnal temperature range (DTR) from daily maximum and minimum temperature data.

    Parameters:
        tmax (xarray.DataArray): Daily maximum temperature data.
        tmin (xarray.DataArray): Daily minimum temperature data.
    Returns:
        xarray.DataArray: Diurnal temperature range data.
    """
    tr = tmax - tmin
    tr = tr.rename('temperature_range')
    return tr


def _consecutive_dry_np(is_dry):
    """NumPy version operating on a 1D time axis."""
    out = np.zeros_like(is_dry, dtype=np.int32)
    count = 0
    for t in range(is_dry.size):
        if is_dry[t]:
            count += 1
            out[t] = count
        else:
            count = 0
    return out


def consecutive_dry_days(precip_data):
    precip_data = precip_data.chunk({"time": -1})
    is_dry = (precip_data < 1).astype("int8")

    consecutive = xr.apply_ufunc(
        _consecutive_dry_np,
        is_dry,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,              # apply independently per grid point
        dask="parallelized",         # works with Dask
        output_dtypes=[np.int32],
    )

    consecutive = consecutive.rename("consecutive_dry_days")
    return consecutive

# def consecutive_dry_days(precip_data):
#     """
#     Calculate consecutive dry days from daily precipitation data.

#     Parameters:
#         precip_data (xarray.DataArray): Daily precipitation data.
#     Returns:
#         xarray.DataArray: Consecutive dry days data.
#     """    
#     # 1 = dry day, 0 = wet day
#     is_dry = (precip_data < 1).astype(int)

#     # Identify boundaries where the dry streak resets
#     reset_points = is_dry == 0

#     # Assign a unique ID to each streak
#     streak_id = reset_points.cumsum(dim='time')

#     # Group by streak_id and compute position within each dry run
#     consecutive = is_dry.groupby(streak_id).cumsum(dim='time')

#     consecutive = consecutive.rename("consecutive_dry_days")
#     return consecutive


def high_rainfall_days(precip_data, percentile=95):
    """
    Find high rainfall days from daily precipitation data.

    Parameters:
        precip_data (xarray.DataArray): Daily precipitation data.
        threshold (float): The precipitation threshold to define a high rainfall day.
    Returns:
        xarray.DataArray: Daily precipitation data for high rainfall days.
    """
    non_zero_precip = precip_data.where(precip_data > 0)
    percentile_data = find_percentile_at_grid(non_zero_precip, 'full_ts', percentile)
    thresholds = percentile_data.sel(dayofyear=non_zero_precip['time.dayofyear'])
    high_rainfall_days = precip_data.where(precip_data > thresholds)
    high_rainfall_days = high_rainfall_days.rename(f'high_precip_{percentile}th')
    high_rainfall_days = xr.merge([high_rainfall_days, precip_data])
    return high_rainfall_days


def find_extreme_days(
        tmax_data, 
        tmin_data, 
        precip_data,
        soil_moisture_data,
        heatwave_threshold, 
        drought_threshold, 
        n_days, 
        threshold_type='percentile', 
        baseline='full_ts',
):
    """
    Find extreme days from temperature and precipitation data.

    Parameters:
        tmax_data (xr.DataArray): Daily maximum temperature data.
        tmin_data (xr.DataArray): Daily minimum temperature data.
        precip_data (xr.DataArray): Daily precipitation data.
        heatwave_threshold (float): The temperature threshold to define a heatwave day.
        drought_threshold (float): The precipitation threshold to define a drought day.
        n_days (list of int): Number of days to find the rolling sum over for drought.
        threshold_type (str): The type of drought threshold to use. Default is 'percentile'.
        baseline (str or list): The baseline to use for the drought percentile. Default is 'full_ts'. Else list of years of form [start_year, end_year].

    Returns:
        xr.Dataset: Dataset with extreme days.
    """
    # heatwave_days = find_heatwave_days(tmax_data, heatwave_threshold)
    # print('heatwave days calculated')
    # heatwave_days.to_netcdf('heatwave_days_just_calced.nc')
    # # print(heatwave_days.sel(time='2000').sum(dim='time').mean())
    # frost_days = find_frost_days(tmin_data)
    # print('forst days calculated')
    # print(frost_days.sel(time='2000').sum(dim='time').mean())
    # save_path = '/g/data/w97/mg5624/ABS_project/extremes/daily/'
    save_path = '/scratch/w97/mg5624/extremes/daily/'
    # new extremes
    # gdd = growing_degree_days(tmax_data, tmin_data, basetemp=0)
    # print('GDD calculated')
    # tr = temperature_range(tmax_data, tmin_data)
    # print('TR calculated')
    # high_rainfall = high_rainfall_days(precip_data, percentile=95)
    # print(high_rainfall)
    # high_rainfall.to_netcdf(save_path + 'high_rainfall.nc')
    # print('high rainfall days calculated')
    cdd = consecutive_dry_days(precip_data)
    # print('CDD calculated')

    # drought_list = []
    # for n in n_days:
    #     met_drought_days = find_drought_days(
    #         precip_data,
    #         'precip',
    #         n, 
    #         drought_threshold, 
    #         threshold_type, 
    #         baseline
    #     )
    #     # print('met drought days claculated')
    #     met_drought_days.to_netcdf(save_path + f'met_drought_days_{n}_days.nc')
    #     # print(met_drought_days.sel(time='2000').sum(dim='time').mean())
    #     drought_list.append(met_drought_days)
        # ag_drought_days = find_drought_days(
        #     soil_moisture_data,
        #     'soil_moisture', 
        #     n,
        #     drought_threshold, 
        #     threshold_type, 
        #     baseline
        # )
        # ag_drought_days.to_netcdf(save_path + f'ag_drought_days_{n}_days.nc')

    #     drought_list.append(ag_drought_days)
    #     print('ag drought days claculated')
    #     print(ag_drought_days.sel(time='2000').sum(dim='time').mean())
    # extremes_list = drought_list + [soil_moisture_data]
    # extremes_list = [precip_data, soil_moisture_data, tmax_data, tmin_data, heatwave_days, frost_days] + drought_list
    # extremes_list = [tmax_data, tmin_data, heatwave_days, frost_days]

    # create extremes ds for new extremes
    # extremes_list = [heatwave_days, frost_days]
    # extremes_list = drought_list
    extremes_list = [cdd]
    # extremes_list = drought_list + [heatwave_days, gdd, tr, high_rainfall, cdd]
    extreme_days = xr.merge(extremes_list)
    # extreme_days = cdd
    # extreme_days = heatwave_days
    return extreme_days


def save_extreme_days(
        years,
        heatwave_threshold,
        drought_threshold,
        n_days,
        threshold_type,
        baseline,
        save_path,
        test=False
):
    """
    Saves the extreme days for the specified years.

    Parameters:
        years (list): List of years to save extreme days for.
        heatwave_threshold (float): The temperature threshold to define a heatwave day.
        drought_threshold (float): The precipitation threshold to define a drought day.
        n_days (int): Number of days to find the rolling sum over for drought.
        threshold_type (str): The type of drought threshold to use. Default is 'percentile'.
        baseline (str or list): The baseline to use for the drought percentile. Default is 'full_ts'. Else list of years of form [start_year, end_year].
        save_path (str): Path to save the extreme days to.

    Returns:
        xr.Dataset: Dataset with extreme days.
    """
    tmax_data = load_multiple_years_of_agcd('tmax', years[0], years[-1], test=test)
    # tmax_data.to_netcdf('tmax_multi_years_just_loaded.nc')
    tmin_data = load_multiple_years_of_agcd('tmin', years[0], years[-1], test=test)
    precip_data = load_multiple_years_of_agcd('precip', years[0], years[-1], test=test)
    sm_data = load_multiple_years_of_agcd('soil_moisture', years[0], years[-1], test=test)
    print('AGCD data loaded')
    # print('tmax: ', tmax_data.isel(time=0).mean())
    # print('tmin: ', tmin_data.isel(time=0).mean())
    # print('hydrometeorological data loaded')
    extreme_days = find_extreme_days(
        tmax_data,
        tmin_data,
        precip_data,
        sm_data,
        heatwave_threshold,
        drought_threshold,
        n_days,
        threshold_type,
        baseline,
    )
    print('extremes dataaray created')
    extreme_days1 = extreme_days.sel(lon=slice(135, None))
    extreme_days2 = extreme_days.sel(lon=slice(None, 135))
    if baseline == 'full_ts':
        baseline = [precip_data['time'].dt.year.min().item(), precip_data['time'].dt.year.max().item()]
    
    drought_threshold = str(drought_threshold).replace('.', 'p')
    heatwave_threshold = str(heatwave_threshold).replace('.', 'p')
    n_days_str = '_'.join([str(n) for n in n_days])
    var = '30_met_drought_centered'
    var = 'cdd'
    # filename1 = f'{var}_{years[0]}-{years[-1]}_1.nc'
    filename2 = f'{var}_{years[0]}-{years[-1]}_2.nc'
    # filename = f'30_met_drought_centered_extremes_{years[0]}-{years[-1]}.nc'
    # filename = f'extreme_days_{years[0]}_{years[-1]}_frost_heatwave_{heatwave_threshold}_met_ag_drought_{n_days_str}_days_{drought_threshold}_{threshold_type}_baseline_{baseline[0]}-{baseline[-1]}.nc'
    # print('--------HEAT EXTREMES---------')
    # filename = 'cdd.nc'
    # filename = f'heatwave_test.nc'
    # filename = 'new_extremes.nc'
    # filename = '30_day_drought_days.nc'
    if test:
        filename = 'test_' + filename
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # extreme_days1.to_netcdf(save_path + filename1)
    extreme_days2.to_netcdf(save_path + filename2)
    # extreme_days.to_netcdf(save_path + filename)
    print('Saved extremes')
    return extreme_days


def main():
    n_days = [
        30,
        # 90, 
        # 180
    ]
    heatwave_threshold = 32
    drought_threshold = 0.15
    threshold_type = 'percentile'
    # save_path = '/g/data/w97/mg5624/ABS_project/extremes/daily/'
    save_path = '/scratch/w97/mg5624/extremes/daily/'
    baseline = 'full_ts'
    test = False
    if test:
        years = [1990, 2021]
    else:
        # years = [1950, 1990]
        years = [1950, 2021]

    save_extreme_days(
        years,
        heatwave_threshold,
        drought_threshold,
        n_days,
        threshold_type,
        baseline,
        save_path,
        test=test
    )


if __name__ == '__main__':
    main()
