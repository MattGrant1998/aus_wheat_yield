import xarray as xr
import re

VARS = [
    "heatwave",
    # "30_day_precip_drought",
    # "90_day_precip_drought",
    "180_day_precip_drought",
    # "30_day_soil_moisture_drought",
    # "90_day_soil_moisture_drought",
    "180_day_soil_moisture_drought",
]


def load_non_filled_ds(var_name, start_year, end_year):
    """Load a non-filled dataset for a given variable and year range."""
    path = f"/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/"
    filename = f"{var_name}_intensity_FY_agg_{start_year}-{end_year}.nc"
    ds = xr.open_dataset(path + filename)
    return ds


def fill_nans(ds):
    ds_filled = ds.fillna(0)
    return ds_filled


def save_filled_ds(ds_filled, var_name, start_year, end_year):
    if 'drought' in var_name:
        match = re.match(r'^(\d+)_day_(.+)$', var_name)
        if match:
            days, rest = match.groups()
            var_name = f'{rest}_{days}'
    path = f"/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/{var_name}/"
    filename = f"{var_name}_intensity_FY_aggregated_{start_year}-{end_year}.nc"
    ds_filled.to_netcdf(path + filename)
    print(f"Saved filled dataset to {path + filename}")


def main():
    start_year = 1950
    end_year = 2021
    for var_name in VARS:
        print(f"Processing {var_name}...")
        ds = load_non_filled_ds(var_name, start_year, end_year)
        ds_filled = fill_nans(ds)
        save_filled_ds(ds_filled, var_name, start_year, end_year)


if __name__ == "__main__":
    main()
