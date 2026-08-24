import numpy as np
import pandas as pd
import xarray as xr
import extremes_functions_on_nc as exf
import multiprocessing
import warnings
import dask
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

def distance_between_days(start_day, end_day, leap_year=False):
    wrap_around = start_day > end_day
    distance = xr.where(
        wrap_around,
        (365 - start_day + 1) + end_day,  # e.g., Dec 31 (365) to Jan 1 (1) is 2 days
        end_day - start_day + 1
    )

    # Adjust for leap year: add 1 if day 60 (Feb 29) is in the range
    if leap_year:
        # Check if 60 falls between start and end, considering wrap
        contains_60 = xr.where(
            wrap_around,
            (start_day <= 60) | (end_day >= 60),
            (start_day <= 60) & (end_day >= 60)
        )
        distance += contains_60.astype(int)
    return distance


def find_extreme_intensities(extremes_ds, extreme):
    extreme_variable = {
        '30_day_soil_moisture_drought': 'soil_moisture_rolling_30_mean',
        '90_day_soil_moisture_drought': 'soil_moisture_rolling_90_mean',
        '180_day_soil_moisture_drought': 'soil_moisture_rolling_180_mean',
        '30_day_precip_drought': 'precip_rolling_30_mean',
        '90_day_precip_drought': 'precip_rolling_90_mean',
        '180_day_precip_drought': 'precip_rolling_180_mean',
        'precip_drought': 'precip',
        'soil_moisture_drought': 'soil_moisture',
        'heatwave': 'tmax',
        'frost': 'tmin'
    }
    climate_var_da = extremes_ds[extreme_variable[extreme]]
    mean_var = exf.find_percentile_at_grid(climate_var_da, 'full_ts', 'mean')
    mean_var = mean_var.sel(dayofyear=extremes_ds['time.dayofyear'])
    # extremes_ds['day_of_year'] = extremes_ds['time'].dt.strftime('%m-%d')
    merged = xr.merge([mean_var, extremes_ds])
    merged[f'{extreme}_intensity'] = abs(
        (merged[f'{extreme_variable[extreme]}_mean'] - merged[extreme]) 
        / merged[f'{extreme_variable[extreme]}_mean']
    ) * 100

    return merged


# def find_extreme_intensities_absolute_thresh(extremes_ds, extreme, threshold):
#     extreme_variable = {
#         'drought': 'precip',
#         'heatwave': 'tmax',
#         'frost': 'tmin'
#     }

#     extreme_var = extreme_variable[extreme]
#     extremes_ds[f'{extreme}_intensity'] = abs(
#         (threshold - extremes_ds[extreme_var]) 
#         / threshold
#     ) * 100

#     return extremes_ds


def add_raw_climate_vars_to_extremes_ds(extremes_ds, climate_vars, precip_agg=90):
    """
    Adds the underlying climate variables to the extremes dataset.

    Args:
        extremes_ds (xarray.Dataset): The dataset containing extreme events.
        climate_vars (list): List of climate variable names to add.
        precip_agg (int): The rolling sum period for precipitation.

    Returns:
        xarray.Dataset: The updated dataset with climate variables added.
    """
    all_years = extremes_ds.coords['time'].dt.year.values
    years = list(np.unique(all_years))
    climate_vars_data = []
    for var in climate_vars:
        var_data = exf.load_multiple_years_of_agcd(var, years[0], years[-1])
        if var == 'precip':
            var_data = var_data.rename({'precip': 'precip_rolling_mean'})
            var_data = var_data.rolling(time=90, center=True).sum()
        climate_vars_data.append(var_data)
    
    extremes_ds_with_climate = xr.merge([extremes_ds] + climate_vars_data)
    return extremes_ds_with_climate


def subset_da_in_time(var_da, start_da, end_da):
    def mask_var_da_time_range(var_da, time, start, end):
        return np.where((time >= start) & (time <= end), var_da, np.nan)
    var_da = var_da.chunk({'time': -1})
    var_da_subset = xr.apply_ufunc(
        mask_var_da_time_range,
        var_da,
        var_da["time"],
        start_da,
        end_da,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[var_da.dtype]
    )

    return var_da_subset


def doy_to_date(x, year):
    x = x.item() if hasattr(x, 'item') else x
    year = year.item() if hasattr(year, 'item') else year
    if np.isnan(x):
        return np.datetime64('NaT', 'D')
    start_of_year = np.datetime64(f"{int(year)}-01-01", 'D')
    return start_of_year + np.timedelta64(int(x) - 1, 'D')


# def doy_to_date(x, year):
#     # Convert x and year to arrays
#     x = np.asarray(x)
#     year = np.asarray(year)

#     # Create output array
#     out = np.full_like(x, np.datetime64('NaT', 'D'), dtype='datetime64[D]')

#     # Mask for valid DOY
#     valid = ~np.isnan(x)

#     # Compute start-of-year for valid entries
#     start = np.array([np.datetime64(f"{int(y)}-01-01", 'D') for y in year[valid]])

#     # Compute dates
#     out[valid] = start + (x[valid].astype(int) - 1).astype("timedelta64[D]")

#     return out

def aggregate_variable(extremes_da, season_stage_doys, end_of_season_doy, agg_type):
    extremes_da['time'] = pd.to_datetime(extremes_da['time'].values)
    agg_var_ds = xr.Dataset()
    start_stage_doy, end_stage_doy = season_stage_doys
    for year in np.unique(extremes_da['time.year'])[1:]:
        end_of_season_date = xr.apply_ufunc(
            doy_to_date,
            end_of_season_doy,
            kwargs={'year': year},
            vectorize=True,
            dask="allowed",
            output_dtypes=[np.dtype("datetime64[D]")]
        )
        # print('END OF SEASON DATE:')
        # print(end_of_season_date)

        end_of_season_month = end_of_season_date.dt.month
        FY_conversion = xr.where(end_of_season_month > 6, year - 1, year)
        
        end_of_season_date_FY = xr.apply_ufunc(
            lambda y, m, d: np.datetime64(
                f"{int(y):04d}-{int(m):02d}-{int(d):02d}", "D"
                ) if not (np.isnan(y) or np.isnan(m) or np.isnan(d)) else np.datetime64('NaT', 'D'),
            FY_conversion,
            end_of_season_date.dt.month,
            end_of_season_date.dt.day,
            vectorize=True,
            dask='allowed',
            output_dtypes=[np.dtype("datetime64[D]")]
        )

        end_stage_year = xr.where(
            end_stage_doy <= end_of_season_doy, 
            end_of_season_date_FY.dt.year, 
            end_of_season_date_FY.dt.year - 1
        )

        end_stage_date = xr.apply_ufunc(
            doy_to_date,
            end_stage_doy, 
            end_stage_year,
            vectorize=True,
            dask="allowed",
            output_dtypes=[np.dtype("datetime64[D]")]
        )

        start_stage_year = xr.where(
            start_stage_doy < end_stage_doy,
            end_stage_date.dt.year,
            end_stage_date.dt.year - 1
        )


        start_stage_date = xr.apply_ufunc(
            doy_to_date,
            start_stage_doy, 
            start_stage_year,
            vectorize=True,
            dask="allowed",
            output_dtypes=[np.dtype("datetime64[D]")]
        )

        # if year == 2007:
        #     print('NSW SEASON DATES:')
        #     print('FY: ', year)
        #     print('start of stage date: ', start_stage_date.sel(lat=-33, lon=148, method='nearest'))
        #     print('end of stage date: ', end_stage_date.sel(lat=-33, lon=148, method='nearest'))
        #     print('end of season date: ', end_of_season_date_FY.sel(lat=-33, lon=148, method='nearest'))

        #     print('QLD SEASON DATES:')
        #     print('FY: ', year)
        #     print('start of stage date: ', start_stage_date.sel(lat=-24, lon=149, method='nearest'))
        #     print('end of stage date: ', end_stage_date.sel(lat=-24, lon=149, method='nearest'))
        #     print('end of season date: ', end_of_season_date_FY.sel(lat=-24, lon=149, method='nearest'))
        extremes_season_stage = subset_da_in_time(extremes_da, start_stage_date, end_stage_date)

        if agg_type == 'sum':
            if extremes_da.name not in ['tmax', 'tmin', 'precip', 'soil_moisture', 'growing_degree_days']:
                extremes_season_stage = xr.where(~np.isnan(extremes_season_stage), 1, 0)
            agg_var = extremes_season_stage.sum(dim='time')
        elif agg_type == 'mean':
            agg_var = extremes_season_stage.mean(dim='time')
        elif agg_type == 'cumsum':
            agg_var = extremes_season_stage.cumsum(dim='time')
        elif agg_type == 'max':
            agg_var = extremes_season_stage.max(dim='time')
        else:
            raise ValueError(f'Aggregation type {agg_type} not recognised.')
        # agg_var = agg_var.expand_dims(time=[np.datetime64(str(year))])
        agg_var = agg_var.expand_dims(time=[year])
        agg_var.name = extremes_da.name
        agg_var = agg_var.to_dataset()
        if not agg_var_ds.data_vars:
            agg_var_ds = agg_var
        else:
            agg_var_ds = xr.concat([agg_var_ds, agg_var], dim='time')

        # print(agg_var)
    # var_name = list(agg_var.data_vars)[0]
    # agg_var[var_name].mean(dim='time').plot()
    # plt.savefig(f'{var_name}_vis.png')
    # print('PLotted 1')
    return agg_var_ds


def circular_midpoint(doy1, doy2, year_len=365):
    """
    Midpoint between two DOYs accounting for year wrap.
    Works with xarray / numpy arrays.
    """
    diff = (doy2 - doy1) % year_len
    mid = (doy1 + diff / 2) % year_len

    # Convert 0 → 365 for DOY convention
    mid = xr.where(mid == 0, year_len, mid)

    return mid


def growing_season_tercile_dates(crop_calendar, antecedent=False):
    # start_of_season = circular_midpoint(
    #     crop_calendar['plant.start'],
    #     crop_calendar['plant']
    # )

    # end_of_season = circular_midpoint(
    #     crop_calendar['harvest'],
    #     crop_calendar['harvest.end']
    # )
    # start_of_season = crop_calendar['plant.start']
    # end_of_season = crop_calendar['harvest.end']
    start_of_season = crop_calendar['plant']
    end_of_season = crop_calendar['harvest']
    if antecedent:
        start_of_ante = start_of_season - 60
        start_of_ante = xr.where(
            start_of_ante <= 0, start_of_ante + 365, start_of_ante
        )
        start_of_ante = xr.where(
            start_of_ante == 0, 365, start_of_ante
        )
        return start_of_ante, start_of_season - 1, end_of_season
    
    mean_season_len = distance_between_days(
        start_of_season,
        end_of_season
    )
    tercile_len = np.floor(mean_season_len / 3)

    # start_of_season = crop_calendar['plant.start']
    start_of_mid = start_of_season + tercile_len + 1
    start_of_harvest = start_of_season + (2 * tercile_len)
    

    start_of_mid = xr.where(
        start_of_mid >= 366, start_of_mid - 365, start_of_mid
    )

    start_of_harvest = xr.where(
        start_of_harvest >= 366, start_of_harvest - 365, start_of_harvest
    )
    
    # Convert 0 to 365 for DOY convention
    end_of_season = xr.where(
        end_of_season < 1, 365, end_of_season
    )

    start_of_harvest = xr.where(
        start_of_harvest < 1, 365, start_of_harvest
    )

    start_of_season = xr.where(
        start_of_season < 1, 365, start_of_season
    )

    start_of_mid = xr.where(
        start_of_mid < 1, 365, start_of_mid
    )

    return start_of_season, start_of_mid, start_of_harvest, end_of_season


def create_ds_of_aggregations_over_growing_season(extremes_da, var, crop, agg_type, grid, antecedent=False, add_antecedent=False, antecedent_days=6*30):
    crop_calendar = load_crop_growing_season_and_regrid(crop, grid)
    if antecedent:
        start_of_ante, end_of_ante, end_of_season = growing_season_tercile_dates(crop_calendar, antecedent=True)
        antecedent_agg = aggregate_variable(extremes_da, [start_of_ante, end_of_ante], end_of_season, agg_type)
        return antecedent_agg
    
    start_of_season, start_of_mid, start_of_harvest, end_of_season = growing_season_tercile_dates(crop_calendar)
    print(start_of_season, start_of_mid, start_of_harvest, end_of_season)
    cropping_dates = xr.merge([
        start_of_season.rename('start_of_season'),
        start_of_mid.rename('start_of_mid'),
        start_of_harvest.rename('start_of_harvest'),
        end_of_season.rename('end_of_season')
    ])
    # cropping_dates.to_netcdf(f'/srv/ccrc/LandAU/z5459957/data/extremes/FY_aggregated/{crop}_growing_season_terciles.nc')

    # import sys
    # sys.exit()
    # start_of_season = crop_calendar['plant.start']
    # start_of_mid = crop_calendar['plant.end'] + 1
    # start_of_harvest = crop_calendar['harvest.start']
    # end_of_season = crop_calendar['harvest.end']

    # print('--------------FULL SEASON ------------------')
    full_growing_agg = aggregate_variable(extremes_da, [start_of_season, end_of_season], end_of_season, agg_type)
    full_growing_agg = full_growing_agg.rename({var: f'growing_seas_{var}_{agg_type}'})

    # print('--------------PLANTING SEASON ------------------')
    planting_agg = aggregate_variable(extremes_da, [start_of_season, start_of_mid - 1], end_of_season, agg_type)
    planting_agg = planting_agg.rename({var: f'planting_seas_{var}_{agg_type}'})

    # print('--------------MID SEASON ------------------')
    mid_season_agg = aggregate_variable(extremes_da, [start_of_mid, start_of_harvest - 1], end_of_season, agg_type)
    mid_season_agg = mid_season_agg.rename({var: f'mid_seas_{var}_{agg_type}'})

    # print('--------------HARVEST SEASON ------------------')
    harvest_agg = aggregate_variable(extremes_da, [start_of_harvest, end_of_season], end_of_season, agg_type)
    harvest_agg = harvest_agg.rename({var: f'harvest_seas_{var}_{agg_type}'})

    full_agg_ds = xr.merge([full_growing_agg, planting_agg, mid_season_agg, harvest_agg])

    if add_antecedent:
        antecedent_first_month = start_of_season - antecedent_days
        if antecedent_first_month < 1:
            antecedent_first_month = antecedent_first_month + 12
        antecedent_agg = aggregate_variable(extremes_da, [antecedent_first_month, start_of_season - 1], end_of_season, agg_type)
        antecedent_agg = antecedent_agg.rename({var: f'{var}_{antecedent_days}_days_before_season_{agg_type}'})
        full_agg_ds = xr.merge([full_agg_ds, antecedent_agg])

    if 'intensity' in var:
        full_agg_ds = full_agg_ds.fillna(0)
    return full_agg_ds


def load_crop_growing_season_and_regrid(crop, target_grid):
    crop_calendar_path = f'/g/data/w97/mg5624/ABS_project/crop_calendar/{crop}.crop.calendar.nc'
    crop_calendar = xr.open_dataset(crop_calendar_path)
    crop_calendar = crop_calendar.sel(latitude=slice(-7, -45), longitude=slice(110, 155))
    crop_calendar = crop_calendar.rename({'latitude': 'lat', 'longitude': 'lon'})
    crop_calendar = crop_calendar.interp_like(target_grid, method='nearest')
    # Fill in missing values
    crop_calendar_filled = crop_calendar.interpolate_na(dim='lat', method='nearest', fill_value='extrapolate')
    crop_calendar_filled = crop_calendar_filled.interpolate_na(dim='lon', method='nearest', fill_value='extrapolate')
    crop_calendar_ls_mask = crop_calendar_filled.where(~np.isnan(target_grid))
    return crop_calendar_ls_mask.load()


def aggregate_variable_for_all_extremes(extremes_ds, variable_args, crop, grid, antecedent=False):
    aggregated_extremes = xr.Dataset()
    for var, args in variable_args.items():
        print(var)
        extremes_da_var = extremes_ds[var].reset_coords(drop=True)
        agg_type, add_antecedent, antecedent_days = args
        season_aggregations = create_ds_of_aggregations_over_growing_season(
            extremes_da_var,
            var,
            crop,
            agg_type,
            grid,
            antecedent,
            add_antecedent,
            antecedent_days
        )
        # aggregated_extremes = xr.merge([aggregated_extremes, season_aggregations])
        # if 'heatwave' in var:
        #     var = f'new_{var}'
        if antecedent:
            filename_out = f'{var}_FY_antecedent_agg_1950-2021.nc'
        else:
            filename_out = f'{var}_FY_agg_1950-2021.nc'
        # filename_out = f'{var}_FY_agg_middling.nc'
        # filename_out = f'{var}_FY_agg_start_end.nc'
        # savefile = 'soil_moisture_aggregations.nc'
        # savefile = 'precip_aggregations_final.nc'
        datadir = '/g/data/w97/mg5624/ABS_project/'
        savepath = datadir + 'extremes/FY_aggregated/'
        season_aggregations.to_netcdf(savepath + filename_out)
    return aggregated_extremes


def find_all_extreme_intensities(extremes_ds):
    underlying_var = {
        '30_day_precip_drought': 'precip_rolling_30_mean',
        '90_day_precip_drought': 'precip_rolling_90_mean',
        '180_day_precip_drought': 'precip_rolling_180_mean',
        '30_day_soil_moisture_drought': 'soil_moisture_rolling_30_mean',
        '90_day_soil_moisture_drought': 'soil_moisture_rolling_90_mean',
        '180_day_soil_moisture_drought': 'soil_moisture_rolling_180_mean',
        'precip_drought': 'precip',
        'soil_moisture_drought': 'soil_moisture',
        'heatwave': 'tmax',
        'frost': 'tmin',
        'high_rainfall_days': 'precip',
    }
    for extreme in extremes_ds.data_vars:
        if 'drought' in extreme or extreme in ['heatwave', 'high_rainfall_days']:
            extremes_ds_var = extremes_ds[[extreme, underlying_var[extreme]]]
            extremes_ds_var = find_extreme_intensities(extremes_ds_var, extreme)
            extremes_ds = xr.merge([extremes_ds, extremes_ds_var])
    return extremes_ds


def main():
    antecedent = False
    heatwave_threshold = 32
    drought_threshold = 0.15
    drought_days = [
        # 30,
        90, 
        # 180
    ]

    baseline = 'full_ts'
    datadir = '/g/data/w97/mg5624/ABS_project/'
    extremes_ds_path = datadir + 'extremes/daily/'
    test = False
    extremes_type = 'soil_moisture'
    
    if test:
        years = [2006, 2010]
    else:
        years = [1950, 2021]

    d_days_label = '_'.join([str(n) for n in drought_days])
    if baseline == 'full_ts':
        baseline_dates = f'{years[0]}-{years[1]}'
    else:
        baseline_dates = f'{baseline[0]}-{baseline[1]}'

    extremes_ds_name = (f'extreme_days_{years[0]}_{years[1]}_frost_heatwave_{heatwave_threshold}'\
                            f'_met_ag_drought_{d_days_label}_days_{str(drought_threshold).replace('.', 'p')}'\
                                f'_percentile_baseline_{baseline_dates}.nc')
    if extremes_type == 'soil_moisture':
        soil_moisture_da = exf.load_multiple_years_of_agcd('soil_moisture', years[0], years[1])
    else:
        soil_moisture_da = None
    filename_dict = {
        'all': extremes_ds_name,
        'temperature': 'temp_extremes_1950-2021.nc',
        'clim': 'mean_clim_1950-2021.nc',
        'drought': 'drought_extremes.nc',
        '180_drought': '180_met_drought_extremes_1950-2022.nc',
        # '180_drought': '180_met_drought_extremes_1950-2022.nc',
        '90_drought': '90_met_drought_extremes_1950-2021.nc',
        '30_drought': '30_met_drought_extremes_1950-2022.nc',
        'met_drought': 'met_drought_extremes.nc', 
        # 'met_drought': 'met_drought_days_30_days.nc',
        '180_ag_drought': '180_ag_drought_extremes_1950-2021.nc',
        'ag_drought': 'ag_drought_extremes.nc',
        # 'ag_drought': 'ag_drought_days_30_days.nc',
        'growing_degree_days': 'gdd_1950-2021.nc',
        'consecutive_dry_days': 'cdd_1950-2021.nc',
        'temperature_range': 'tr_1950-2021.nc',
        'soil_moisture': soil_moisture_da,
        'test': 'old_versions/test_extreme_days_2006_2010_frost_heatwave_32_met_ag_drought_90_days_0p15_percentile_baseline_2006-2010.nc',
    }
    # if test:
    #     extremes_ds_name = 'test_extreme_days_2006_2010_frost_heatwave_32_met_ag_drought_90_days_0p15_percentile_baseline_2006-2010.nc'
    # else:
    #     extremes_ds_name = (f'extreme_days_{years[0]}_{years[1]}_frost_heatwave_{heatwave_threshold}'\
    #                         f'_met_ag_drought_{d_days_label}_days_{str(drought_threshold).replace('.', 'p')}_percentile_baseline_{baseline_dates}.nc')
    if extremes_type == 'soil_moisture':
        extremes_da = filename_dict[extremes_type]
        extremes_ds = extremes_da.to_dataset()
    else:
        filename_in = filename_dict[extremes_type]
        if test:
            filename_in = 'test_' + filename_in
    
        extremes_ds = xr.open_dataset(extremes_ds_path + filename_in)
        extremes_ds = extremes_ds.reset_coords(drop=True)
    agcd_filepath = '/g/data/zv2/agcd/v1-0-1/precip/total/r005/01day/'
    grid = xr.open_dataset(agcd_filepath + 'agcd_v1-0-1_precip_total_r005_daily_2016.nc').precip.isel(time=-50)
    grid = grid.reset_coords(drop=True)

    extremes_ds = extremes_ds.chunk(lat=50, lon=50)
    extremes_ds['time'] = pd.to_datetime(extremes_ds['time'].values)
    # extremes_ds_with_climate_vars = add_raw_climate_vars_to_extremes_ds(extremes_ds, ['precip', 'tmax', 'tmin'])

    extremes_ds = find_all_extreme_intensities(extremes_ds)

    variable_args = {
        'precip': ['mean', False, None],
        'tmax': ['mean', False, None],
        'tmin': ['mean', False, None],
        'tavg': ['mean', False, None],
        'soil_moisture': ['mean', False, None],
        '90_day_precip_drought': ['sum', False, None],
        '90_day_soil_moisture_drought': ['sum', False, None],
        '180_day_precip_drought': ['sum', False, None],
        '180_day_soil_moisture_drought': ['sum', False, None],
        'heatwave': ['sum', False, None],
        'frost': ['sum', False, None],
        'precip_drought_intensity': ['mean', False, None],
        'soil_moisture_drought_intensity': ['mean', False, None],
        'heatwave_intensity': ['mean', False, None],
        'frost_intensity': ['mean', False, None],
        'growing_degree_days': ['cumsum', False, None],
        'consecutive_dry_days': ['max', False, None],
        'temperature_range': ['mean', False, None],
        'high_rainfall_days': ['sum', False, None],
        'high_rainfall_intensity': ['mean', False, None],
    }

    drought_variable_args = {
        'precip': ['mean', False, None],
        'soil_moisture': ['mean', False, None],
        '90_day_precip_drought': ['sum', False, None],
        '90_day_soil_moisture_drought': ['sum', False, None],
        '180_day_precip_drought': ['sum', False, None],
        '180_day_soil_moisture_drought': ['sum', False, None],
        'precip_drought_intensity': ['mean', False, None],
        'soil_moisture_drought_intensity': ['mean', False, None],
    }

    drought_180_variable_args = {
        # 'precip': ['mean', False, None],
        '180_day_precip_drought_intensity': ['mean', False, None],
        '180_day_precip_drought': ['sum', False, None],
    }

    drought_90_variable_args = {
        # 'precip': ['mean', False, None],
        '90_day_precip_drought_intensity': ['mean', False, None],
        '90_day_precip_drought': ['sum', False, None],
    }

    drought_30_variable_args = {
        # 'precip': ['mean', False, None],
        '30_day_precip_drought_intensity': ['mean', False, None],
        '30_day_precip_drought': ['sum', False, None],
    }

    met_drought_variable_args = {
        'precip': ['mean', False, None],
        '30_day_precip_drought': ['sum', False, None],
        '90_day_precip_drought': ['sum', False, None],
        '180_day_precip_drought': ['sum', False, None],
        '30_day_precip_drought_intensity': ['mean', False, None],
        '90_day_precip_drought_intensity': ['mean', False, None],
        '180_day_precip_drought_intensity': ['mean', False, None],
        'consecutive_dry_days': ['max', False, None],
    }

    ag_drought_variable_args = {
        # 'soil_moisture': ['mean', False, None],
        # '30_day_soil_moisture_drought': ['sum', False, None],
        # '90_day_soil_moisture_drought': ['sum', False, None],
        '180_day_soil_moisture_drought': ['sum', False, None],
        # '30_day_soil_moisture_drought_intensity': ['mean', False, None],
        # '90_day_soil_moisture_drought_intensity': ['mean', False, None],
        '180_day_soil_moisture_drought_intensity': ['mean', False, None],
    }

    ag_drought_180_variable_args = {
        '180_day_soil_moisture_drought_intensity': ['mean', False, None],
        '180_day_soil_moisture_drought': ['sum', False, None],
    }

    temp_variable_args = {
        # 'tmax': ['mean', False, None],
        # 'tmin': ['mean', False, None],
        # 'temperature_range': ['mean', False, None],
        # 'growing_degree_days': ['sum', False, None],
        'heatwave': ['sum', False, None],
        # 'frost': ['sum', False, None],
        'heatwave_intensity': ['mean', False, None],
        # 'frost_intensity': ['mean', False, None],
    }

    test_args = {
        'precip': ['sum', False, None],
        'tmax': ['mean', False, None],
        'tmin': ['mean', False, None],
        'soil_moisture': ['mean', False, None],
        'heatwave': ['sum', False, None],
        'frost': ['sum', False, None],
        'precip_drought': ['sum', False, None],
        'soil_moisture_drought': ['sum', False, None],
        'precip_drought_intensity': ['mean', False, None],
        'soil_moisture_drought_intensity': ['mean', False, None],
    }
    # vars = 'drought'
    # vars = 'temp'
    # vars = 'soil_moisture'
    # vars = 'all'
    mean_clim_args = {
        # 'precip': ['mean', False, None],
        # 'tmax': ['mean', False, None],
        # 'tmin': ['mean', False, None],
        # 'tavg': ['mean', False, None],
        'growing_degree_days': ['cumsum', False, None],
        # 'consecutive_dry_days': ['max', False, None],
        # 'temperature_range': ['mean', False, None],
        # 'soil_moisture': ['mean', False, None],
    }

    args = {
        'all': variable_args,
        'growing_degree_days': {'growing_degree_days': ['sum', False, None]},
        'temperature_range': {'temperature_range': ['mean', False, None]},
        'consecutive_dry_days': {'consecutive_dry_days': ['max', False, None]},
        'soil_moisture': {'soil_moisture': ['mean', False, None]},
        'drought': drought_variable_args,
        '180_drought': drought_180_variable_args,
        '90_drought': drought_90_variable_args,
        '30_drought': drought_30_variable_args,
        '180_ag_drought': ag_drought_180_variable_args,
        'met_drought': met_drought_variable_args,
        'ag_drought': ag_drought_variable_args,
        'temperature': temp_variable_args,
        'clim': mean_clim_args,
        'test': test_args
    }

    aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, args[extremes_type], 'Wheat', grid, antecedent)
    # if vars == 'drought':
    #     aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, drought_variable_args, 'Wheat', grid)
    # elif vars == 'temp':
    #     aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, temp_variable_args, 'Wheat', grid)
    # elif vars == 'precip_drought':
    #     aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, met_drought_variable_args, 'Wheat', grid)
    # elif vars == 'soil_moisture_drought':
    #     aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, ag_drought_variable_args, 'Wheat', grid)
    # else:
    #     aggregated_extremes = aggregate_variable_for_all_extremes(extremes_ds, variable_args, 'Wheat', grid)
    savepath = datadir + 'extremes/FY_aggregated/'
    # savefile = (f'{vars}_agg_extremes_{years[0]}_{years[1]}_frost_heatwave_{heatwave_threshold}'\
    #             f'_drought_{d_days_label}_days_{str(drought_threshold).replace('.', 'p')}_percentile_baseline_{baseline_dates}.nc')
    
    # filename_out = filename_in.replace('.nc', f'_FY_agg.nc')
    # savefile = 'soil_moisture_aggregations.nc'
    # savefile = 'precip_aggregations_final.nc'
    # if test:
    #     savefile = 'test_' + savefile
    # filename_out = 'test_' + filename_out
    # aggregated_extremes.to_netcdf(savepath + filename_out)


if __name__ == "__main__":
    main()
