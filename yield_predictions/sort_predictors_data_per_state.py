import xarray as xr
import pandas as pd
import os

STATE_MASKS = {
    # 'NSW': 1,
    # 'VIC': 2,
    # 'QLD': 3,
    # 'SA': 4,
    'WA': 5
}

PREDICTORS = [
    '180_day_precip_drought',
    '180_day_precip_drought_intensity',
    '180_day_soil_moisture_drought',
    '180_day_soil_moisture_drought_intensity',
    'consecutive_dry_days',
    'frost',
    'growing_degree_days',
    'heatwave',
    'heatwave_intensity',
    'precip',
    'soil_moisture',
    'temperature_range',
    'tmax',
    'tmin',
]

PREDICTOR_AGG = {
    '180_day_precip_drought': 'sum',
    '180_day_precip_drought_intensity': 'mean',
    '180_day_soil_moisture_drought': 'sum',
    '180_day_soil_moisture_drought_intensity': 'mean',
    'consecutive_dry_days': 'max',
    'frost': 'sum',
    'growing_degree_days': 'sum',
    'heatwave': 'sum',
    'heatwave_intensity': 'mean',
    'precip': 'mean',
    'soil_moisture': 'mean',
    'temperature_range': 'mean',
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


def load_predictors_data(state):
    """
    Load the predictors data for a specific state from a NetCDF file.

    Args:
        state (str): The state name (e.g., 'NSW', 'VIC', 'QLD', 'SA', 'WA').
    Returns:
        xarray.Dataset: The predictors dataset for the specified state.
    """
    state_mask_value = STATE_MASKS[state]
    state_mask = load_state_mask()
    state_masked = state_mask.where(state_mask == state_mask_value)

    intensity_predictors = [
        '180_day_precip_drought_intensity',
        '180_day_soil_moisture_drought_intensity',
        'heatwave_intensity'
    ]
    predictors_data = {}
    for predictor in PREDICTORS:
        path = f'/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/{predictor}_FY_agg_1950-2021.nc'
        agg = PREDICTOR_AGG[predictor]
        predictor_data = xr.open_dataset(path)[f'growing_seas_{predictor}_{agg}']
        if predictor in intensity_predictors:
            predictor_data = predictor_data.fillna(0)
        predictor_data_state = predictor_data.where(state_masked.notnull())
        predictors_data[f'growing_seas_{predictor}_{agg}'] = predictor_data_state
    predictors_dataset = xr.Dataset(predictors_data)
    return predictors_dataset


def save_predictors_data_per_state(state, predictors_dataset):
    """
    Save the predictors dataset for a specific state to a NetCDF file.

    Args:
        state (str): The state name (e.g., 'NSW', 'VIC', 'QLD', 'SA', 'WA').
        predictors_dataset (xarray.Dataset): The predictors dataset for the specified state.
    """
    output_path = f'/g/data/w97/mg5624/ABS_project/predictors_per_state/{state}_predictors.nc'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    predictors_dataset.to_netcdf(output_path)


def main():
    for state in STATE_MASKS.keys():
        print(f"Processing predictors for state: {state}")
        predictors_dataset = load_predictors_data(state)
        save_predictors_data_per_state(state, predictors_dataset)
        print(f"Saved predictors dataset for state: {state}")


if __name__ == "__main__":
    main()
