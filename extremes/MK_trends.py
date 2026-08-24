import xarray as xr
import warnings
import os
import multiprocessing
import pandas as pd
import pymannkendall as mk
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning) 

datadir = '/g/data/w97/mg5624/ABS_project/'


def aggregate_drought_events_per_time_period(drought_events_data, number_of_years, year_from, year_to, agg_type='sum'):
    """
    Finds the number of drought events per specified number of year for each grid cell.

    Args:
        drought_events_data (xr.DataArray): drought event data (monthly timescale)
        number_of_years (int): number of years to sum the number of droughts over
        year_from (int or str): year to take trend from
        year_to (int or str): year to take trend to
        agg_type (optional: 'mean' or 'sum'): whether to sum or mean over the time period

    Returns:
        drought_events_per_year (xr.DataArray): number of events per year
    """
    # This line resamples the data to be blocked sums over every 'number_of_years'. It starts from the first year of the data
    # and will end once all the years are included in one of the blocks. This could mean that the final block includes years 
    # for which we have no data for - the if conditions below sort this out
    # aggregated_metric = drought_events_data.resample(time=f'{number_of_years}Y', closed='right', label='left').sum(dim='time')

    # Sum number of drought events of blocks of the number of years specified. 
    # year_block here assigns each timestep to it's year block e.g. for data starting in 1911, any data in the years 
    # 1911, 1912, 1913, 1914, and 1915 will be assigned to the 1911 year_block to be summed together later
    year_block = (drought_events_data['time.year'].data - int(year_from)) // int(number_of_years) * int(number_of_years) + int(year_from)
    drought_events_data.coords['year_block'] = ('time', year_block)

    # Group by 'year_block' and sum over 'time'
    ds_groups = drought_events_data.groupby('year_block')
    if agg_type == 'sum':
        aggregated_metric = ds_groups.sum('time')
    elif agg_type == 'mean':
        aggregated_metric = ds_groups.mean('time')
    else:
        raise ValueError(f'aggregation type {agg_type} not valid for this function, use \'mean\' or \'sum\'.')
    aggregated_metric = aggregated_metric.rename({'year_block': 'time'})

    # Checks to ensure the block summing has worked as intended
    time = aggregated_metric['time'].values

    data_start_year = int(time[0])
    data_end_year = int(time[-1]) #converting to int gives it as years since 1970

    # These ensure we aren't including years in our blocked resampled data which we don't have any data for
    if data_start_year < int(year_from):
        aggregated_metric = aggregated_metric.isel(time=slice(1, None))
    # minus the one to number_of_years below, as year_blocks are closed on the right, e.g. 2016-2020 includes 2020
    if data_end_year > (int(year_to) - (number_of_years - 1)): 
        aggregated_metric = aggregated_metric.isel(time=slice(None, -1))
    return aggregated_metric


def regrid_mask(mask, grid):
    """
    Regrids a given mask to match the grid of another dataset.

    Parameters:
    mask (xarray.DataArray): The mask to be regridded. It should have latitude ('lat') as one of its dimensions.
    grid (xarray.DataArray): The target grid to which the mask will be regridded.

    Returns:
    xarray.DataArray: The regridded mask with the same dimensions as the input grid, converted to boolean type.
    """
    mask = mask.sortby('lat')
    mask = mask.astype(int)
    regridded_mask = mask.interp_like(grid, method='nearest')

    # Build bounding-box mask
    lat_min = float(mask.lat.min())
    lat_max = float(mask.lat.max())
    lon_min = float(mask.lon.min())
    lon_max = float(mask.lon.max())

    # Boolean mask: True where grid is inside original mask domain
    inside = (
        (grid.lat >= lat_min) & (grid.lat <= lat_max) &
        (grid.lon >= lon_min) & (grid.lon <= lon_max)
    )

    # Force outside-domain points to False
    regridded_mask = regridded_mask.where(inside, 0)
    new_mask = regridded_mask.astype(bool)

    new_mask = regridded_mask.astype(bool)

    return new_mask


def calculate_trendtest(data, test_type, year_from, year_to, events=False, aggregate=None):
    """
    Computes the MK or LogReg trendtest for each grid point in the data.

    Args:
        data (xr.DataArray): spatial and temporal data
        test_type (str): MK trend test type being performed or LogReg for logistic regression
        year_from (int): year from whcih the trend goes from
        year_to (int): year to which the trend goes to
        events (bool): if calculating for events set to True
        aggregate_years (int or str): if events=True, number of years to sum events over or 'LogReg' if using logistic regression

    Returns:
        MK_da (xr.DataArray): trend test result at each grid point
    """
    data = data.sel(time=slice(year_from, year_to))

    mask = xr.open_dataarray(f'/g/data/w97/mg5624/RF_project/masks/regridded_awra_awap_mask.nc')
    mask = regrid_mask(mask, data)
    data_masked = data.where(mask)
    
    if aggregate is not None:
        data_masked = aggregate_drought_events_per_time_period(data_masked, aggregate, year_from, year_to)
    trend_df = pd.DataFrame()
    print('Start calculating MK trend')
    for i in data_masked['lat'].values:
        for j in data_masked['lon'].values:
            data_ij = data_masked.sel(lat=i, lon=j).values
            # data_ij_df = data_ij.to_dataframe()
            # data_ij_df.reset_index(inplace=True)
            # if all values are nan, then skip this grid point
            if np.any(np.isnan(data_ij)):
                trend_dict = {'lat': i, 'lon': j, 'MK_trend': np.nan, 'MK_slope': np.nan}
                # break
            else:
                if test_type == 'MK_original':
                    trend_result = mk.original_test(data_ij)
                elif test_type == 'MK_hamed_rao':
                    trend_result = mk.hamed_rao_modification_test(data_ij)
                elif test_type == 'MK_yue_wang':
                    trend_result = mk.yue_wang_modification_test(data_ij)
                elif test_type == 'MK_seasonal_sens_slope':
                    trend_result = mk.seasonal_sens_slope(data_ij)
                if test_type == 'MK_seasonal_sens_slope':
                    trend_dict = {'lat': i, 'lon': j, 'MK_slope': trend_result.slope}
                else:
                    trend_dict = {'lat': i, 'lon': j, 'MK_trend': trend_result.trend, 'MK_slope': trend_result.slope}
            trend_df_ij = pd.DataFrame([trend_dict])
            trend_df = pd.concat((trend_df, trend_df_ij))

    if 'MK_trend' in trend_df.columns:
        trend_df['MK_trend'].replace({'no trend': 0, 'decreasing': -1, 'increasing': 1}, inplace=True)
    
    trend_df.set_index(['lat', 'lon'], inplace=True)
    trend_da = trend_df.to_xarray()
    
    return trend_da


def load_and_save_data(variable, year_from, year_to, season_stage, test_type='MK_original', aggregate=None, replace=False):
    """
    Loads the data array of the hydrometeorological variable, calculates the MK trend test, and saves the results.

    Args:
        hydromet_variable (str): Name of the hydrometeorological variable to load.
        year_from (int): Year from which the trend analysis starts.
        year_to (int): Year to which the trend analysis ends.
        season (str, optional): If requiring seasonal data, specify the season (e.g., 'DJF'), else None.
        test_type (str, optional): Type of MK test to be conducted ('MK_original', 'MK_hamed_rao', 'MK_yue_wang', 'seasonal_sens_slope', 'LogReg').
        replace (bool, optional): If True, replace existing files.

    Returns:
        MK_data (xr.DataArray): MK trend test results for the specified variable.
    """
    # final_year = {
    #     '180_day_precip_drought_sum': 2021,
    #     '90_day_precip_drought_sum': 2021,
    #     '30_day_precip_drought_um': 2021,
    #     'heatwave': 2021,
    #     'frost': 2021,
    # }
    variable_label = None
    final_year = 2021
    if 'sum' in variable:
        variable_label = variable[:-4]
    elif 'intensity' in variable:
        variable_label = variable[:-5]

    if variable_label is not None:
        data_files_in = datadir + f'/extremes/FY_aggregated/{variable_label}_FY_agg_1950-{final_year}.nc'
    elif 'antecedent' in variable:
        data_files_in = datadir + f'/extremes/FY_aggregated/{variable}_1950-{final_year}.nc'
    if season_stage == 'antecedent':
        data = xr.open_dataarray(data_files_in)
    else:
        data = xr.open_dataset(data_files_in)[f'{season_stage}_seas_{variable}']
    data = data.fillna(0)

    mask = xr.open_dataarray(f'/g/data/w97/mg5624/RF_project/masks/regridded_awra_awap_mask.nc')
    
    mask = regrid_mask(mask, data)
    data = data.where(mask)

    filepath_out = datadir + f'extremes/MK_test/{season_stage}/{variable}/'
    filename_out = f'{year_from}-{year_to}_{test_type}_{variable}.nc'
    if aggregate is not None:
        filename_out = f'{year_from}-{year_to}_{test_type}_{variable}_agg_{aggregate}yr.nc'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    
    if os.path.exists(filepath_out + filename_out) and not replace:
        print('passing')
        pass
    else:
        MK_data = calculate_trendtest(data, test_type, year_from, year_to, aggregate)
        MK_data.to_netcdf(filepath_out + filename_out)
    return MK_data


def run_MK_test_for_variables():
    arguments = []                  
    start_year = '1970'
    end_year = '2020'
    VARIABLES = [
        '30_day_precip_drought_FY_antecedent_agg',
        # '180_day_precip_drought_sum', 
        # '90_day_precip_drought_sum', 
        # '30_day_precip_drought_sum', 
        # 'heatwave_sum', 
        # 'frost_sum',
        # '180_day_precip_drought_intensity_mean', 
        # '90_day_precip_drought_intensity_mean', 
        # '30_day_precip_drought_intensity_mean', 
        # 'heatwave_intensity_mean', 
    ]
    for season in [
        # 'growing', 
        # 'planting', 
        # 'mid', 
        # 'harvest',
        'antecedent',
        ]:
        for var in VARIABLES:
            arguments.append((
                var, 
                start_year, 
                end_year, 
                season, 
                'MK_yue_wang', 
                None, #aggregate
                True, #replace
            ))

    nb_cpus = min(multiprocessing.cpu_count(), len(arguments))
    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(load_and_save_data, arguments)
    pool.close()
    pool.join()


def main():
    run_MK_test_for_variables()


if __name__ == '__main__':
    main()
    