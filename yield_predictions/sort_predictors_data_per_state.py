import xarray as xr
import pandas as pd
import os

STATE_MASKS = {
    'NSW': 1,
    'VIC': 2,
    'QLD': 3,
    'SA': 4,
    'WA': 5
}

PREDICTORS = [
    'precip_drought_180',
    'precip_drought_180_intensity',
    'sm_drought_180',
    'sm_drought_180_intensity',
    'cdd',
    'frost',
    'gdd',
    'heatwave',
    'heatwave_intensity',
    'precip',
    'sm',
    'tr',
    'tmax',
    'tmin',
]

PREDICTOR_AGG = {
    'precip_drought_180': 'sum',
    'precip_drought_180_intensity': 'mean',
    'sm_drought_180': 'sum',
    'sm_drought_180_intensity': 'mean',
    'cdd': 'max',
    'frost': 'sum',
    'gdd': 'sum',
    'heatwave': 'sum',
    'heatwave_intensity': 'mean',
    'precip': 'mean',
    'sm': 'mean',
    'tr': 'mean',
    'tmax': 'mean',
    'tmin': 'mean'
}


def load_state_mask():
    """
    Load the state mask data from a NetCDF file.

    Returns:
        xarray.DataArray: The state mask data array.
    """
    path = '/g/data/w97/mg5624/ABS_project/masks/state_wheat_mask.nc'
    state_mask = xr.open_dataarray(path)
    return state_mask


def load_predictors_data(state, start_year, end_year):
    """
    Load the predictors data for a specific state from a NetCDF file.

    Args:
        state (str): The state name (e.g., 'NSW', 'VIC', 'QLD', 'SA', 'WA').
        start_year (int): The starting year for the data range.
        end_year (int): The ending year for the data range.

    Returns:
        xarray.Dataset: The predictors dataset for the specified state.
    """
    state_mask_value = STATE_MASKS[state]
    state_mask = load_state_mask()
    state_masked = state_mask.where(state_mask == state_mask_value)

    intensity_predictors = [
        'precip_drought_180_intensity',
        'sm_drought_180_intensity',
        'heatwave_intensity'
    ]
    predictors_data = {}
    for predictor in PREDICTORS:
        path = f'/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/{predictor}/{predictor}_FY_aggregated_{start_year}-{end_year}.nc'
        agg = PREDICTOR_AGG[predictor]
        predictor_data = xr.open_dataset(path)[f'growing_seas_{predictor}_{agg}']
        if predictor in intensity_predictors:
            predictor_data = predictor_data.fillna(0)
        predictor_data_state = predictor_data.where(state_masked.notnull())
        predictors_data[f'growing_seas_{predictor}_{agg}'] = predictor_data_state
    predictors_dataset = xr.Dataset(predictors_data)
    return predictors_dataset


def save_predictors_data_per_state(state, predictors_dataset, start_year, end_year):
    """
    Save the predictors dataset for a specific state to a NetCDF file.

    Args:
        state (str): The state name (e.g., 'NSW', 'VIC', 'QLD', 'SA', 'WA').
        predictors_dataset (xarray.Dataset): The predictors dataset for the specified state.
        start_year (int): The starting year for the data range.
        end_year (int): The ending year for the data range.
    """
    output_path = f'/g/data/w97/mg5624/ABS_project/predictors_per_state/{state}_predictors_{start_year}-{end_year}.nc'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    predictors_dataset.to_netcdf(output_path)


def main():
    start_year = 2022
    end_year = 2025
    for state in STATE_MASKS.keys():
        print(f"Processing predictors for state: {state}")
        predictors_dataset = load_predictors_data(state, start_year, end_year)
        save_predictors_data_per_state(state, predictors_dataset, start_year, end_year)
        print(f"Saved predictors dataset for state: {state}")


if __name__ == "__main__":
    main()
