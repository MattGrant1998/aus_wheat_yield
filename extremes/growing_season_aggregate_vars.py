"""
aggregate_obs_extremes.py

Aggregate the AGCD/AWRA-L-derived daily agroclimatic extremes (produced by
obs_indices.py) over the wheat growing season -- the obs equivalent of
aggregate_narclim_extremes.py. Runs through every variable in VAR_CONFIG
in one pass instead of needing manual per-variable tweaking.
"""

import sys
from pathlib import Path
import warnings

import xarray as xr
import os
warnings.simplefilter("ignore")

# =====================================================================
# IMPORTS FROM YOUR EXISTING MODULES -- adjust these paths as needed
# =====================================================================

sys.path.append(str(Path("/home/561/mg5624/yield_trends/extremes")))
from extremes_characteristics_nc import create_ds_of_aggregations_over_growing_season

# obs_indices.py from the daily-extremes step -- reused here for its
# config (OUT_ROOT, START_YEAR, END_YEAR, NC_VAR_NAME) and its opener /
# rolling / percentile functions, so file paths and variable names stay
# in sync between the two scripts automatically.
from daily_extremes import (
    OUT_ROOT, NC_VAR_NAME,
    open_agcd_var, open_awra_sm, rolling_data, find_percentile_at_grid,
)

DAILY_ROOT = OUT_ROOT
OUT_ROOT = '/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/'

START_YEAR = 2021
END_YEAR = 2025

CROP = "Wheat"

# =====================================================================
# CONFIG -- add/remove a variable here and it's automatically picked up
# by the loop in main(). No other code needs to change.
# =====================================================================

# var_name -> (agg_type for the event/variable itself, has_intensity)
VAR_CONFIG = {
    # "heatwave": ("sum", True),
    # "frost": ("sum", False),
    # "gdd": ("sum", False),
    # "tr": ("mean", False),
    # "cdd": ("max", False),
    # "tmax": ("mean", False),
    # "tmin": ("mean", False),
    # "precip": ("mean", False),
    # "sm": ("mean", False),
    # # "precip_drought_30": ("sum", True),
    # # "precip_drought_90": ("sum", True),
    # "precip_drought_180": ("sum", True),
    # "sm_drought_30": ("sum", True),
    # "sm_drought_90": ("sum", True),
    "sm_drought_180": ("sum", True),
}

# extreme var_name -> which raw underlying variable its intensity is
# computed against (key into NC_VAR_NAME / RAW_VAR_OPENERS below)
UNDERLYING_VAR = {
    "heatwave": "tmax",
    "precip_drought_30": "precip",
    "precip_drought_90": "precip",
    "precip_drought_180": "precip",
    "sm_drought_30": "sm",
    "sm_drought_90": "sm",
    "sm_drought_180": "sm",
}

# Variables in VAR_CONFIG that are raw AGCD/AWRA-L variables rather than
# daily-extremes files saved by obs_indices.py (those never get written
# to disk by that script -- it only saves the derived indices).
RAW_VAR_OPENERS = {
    "tmax": lambda: open_agcd_var("tmax", START_YEAR, END_YEAR),
    "tmin": lambda: open_agcd_var("tmin", START_YEAR, END_YEAR),
    "precip": lambda: open_agcd_var("precip", START_YEAR, END_YEAR),
    "sm": lambda: open_awra_sm(START_YEAR, END_YEAR),
}

_RAW_CACHE = {}


# =====================================================================
# Opening data
# =====================================================================

def get_raw_da(var_key):
    """Load (and cache) a raw underlying variable as a plainly-named DataArray."""
    if var_key not in _RAW_CACHE:
        ds = RAW_VAR_OPENERS[var_key]()
        if ds is None:
            _RAW_CACHE[var_key] = None
        elif isinstance(ds, xr.DataArray):
            da = ds.reset_coords(drop=True)
            da.name = var_key
            _RAW_CACHE[var_key] = da
        elif isinstance(ds, xr.Dataset):
            da = ds[NC_VAR_NAME[var_key]].reset_coords(drop=True)
            da.name = var_key
            _RAW_CACHE[var_key] = da
        else:
            raise TypeError(
                f"RAW_VAR_OPENERS['{var_key}'] returned {type(ds)!r}, "
                "expected xarray.DataArray or xarray.Dataset"
            )
    return _RAW_CACHE[var_key]


def get_reference_grid():
    """2D lat/lon template used by the growing-season aggregator."""
    tmax = get_raw_da("tmax")
    if tmax is None:
        raise RuntimeError("Could not load AGCD tmax to build a reference grid")
    return tmax.isel(time=0).reset_coords(drop=True)


def open_derived_var(var_name):
    """Open a daily extremes file previously saved by obs_indices.py."""
    path = DAILY_ROOT + f"{var_name}/{var_name}_{START_YEAR}-{END_YEAR}.nc"
    if not os.path.exists(path):
        print(f"  !! {path} not found")
        return None
    ds = xr.open_dataset(path)
    da = list(ds.data_vars.values())[0].reset_coords(drop=True)
    da.name = var_name
    return da


def open_event_da(var_name):
    """Open the daily DataArray for `var_name`, whichever kind it is."""
    if var_name in RAW_VAR_OPENERS:
        return get_raw_da(var_name)
    return open_derived_var(var_name)


# =====================================================================
# Saving
# =====================================================================

def save_growing_season_output(var, ds_out):
    out_dir = OUT_ROOT + f"{var}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + f"/{var}_FY_aggregated_{START_YEAR}-{END_YEAR}.nc"
    ds_out.to_netcdf(out_path)
    print(f"  Saved -> {out_path}")


def save_intensity(var, da):
    int_name = f"{var}_intensity"
    out_dir = OUT_ROOT + int_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = out_dir + f"/{int_name}_FY_aggregated_{START_YEAR}-{END_YEAR}.nc"
    da.to_netcdf(out_path)
    print(f"  Saved intensity -> {out_path}")


def compute_intensity(extreme_da, extreme_name):
    if extreme_name not in UNDERLYING_VAR:
        return None

    underlying_key = UNDERLYING_VAR[extreme_name]
    raw_da = get_raw_da(underlying_key)
    if raw_da is None:
        print(f"  !! Underlying var {underlying_key} not available for {extreme_name}")
        return None

    # Drought indices are themselves N-day rolling means, so the
    # climatology compared against must be the same rolling window --
    # not the raw daily value. Window size is read off the variable name
    # (e.g. "precip_drought_30" -> 30).
    if extreme_name.rsplit("_", 1)[-1].isdigit():
        window = int(extreme_name.rsplit("_", 1)[-1])
        reference_da = rolling_data(raw_da, window, "mean")
    else:
        reference_da = raw_da

    mean_var = find_percentile_at_grid(reference_da, "mean")
    mean_var = mean_var.sel(dayofyear=extreme_da["time.dayofyear"])

    intensity = (abs(mean_var - extreme_da) / mean_var) * 100
    intensity = intensity.rename(f"{extreme_name}_intensity")
    return intensity


# =====================================================================
# Main
# =====================================================================

def main():
    grid = get_reference_grid()

    for var, (agg_type, has_intensity) in VAR_CONFIG.items():
        print(f"\n=== {var} ===")

        # ---------- aggregate the event/variable itself ----------
        out_dir = OUT_ROOT + f"{var}"
        out_path = out_dir + f"/{var}_FY_aggregated_{START_YEAR}-{END_YEAR}.nc"
        if os.path.exists(out_path):
            print(f"  [SKIP] {var} FY aggregation already exists -> {out_path}")
        else:
            da = open_event_da(var)
            if da is None:
                continue
            print(f"  -> Aggregating {var} with agg_type='{agg_type}'")
            da = da.load()  # ensure fully loaded before the per-year season slicing
            ds_out = create_ds_of_aggregations_over_growing_season(
                da, var, CROP, agg_type, grid,
                add_antecedent=False, antecedent_days=None,
            )
            save_growing_season_output(var, ds_out)

        # ---------- compute and aggregate intensity ----------
        if has_intensity and var in UNDERLYING_VAR:
            int_name = f"{var}_intensity"
            int_FY_path = (
                OUT_ROOT + f"/{int_name}/"
                + f"{int_name}_FY_aggregated_{START_YEAR}-{END_YEAR}.nc"
            )
            if os.path.exists(int_FY_path):
                print(f"  [SKIP] {int_name} FY aggregation already exists -> {int_FY_path}")
                continue

            int_daily_path = DAILY_ROOT + f"/{int_name}/{int_name}_{START_YEAR}-{END_YEAR}.nc"
            if os.path.exists(int_daily_path):
                ds_int = xr.open_dataset(int_daily_path)
                da_int = list(ds_int.data_vars.values())[0].reset_coords(drop=True)
                da_int.name = int_name
                print(f"  -> Using existing daily intensity {int_name}")
            else:
                da_event = open_event_da(var)
                if da_event is None:
                    print(f"  !! Cannot compute intensity: {var} not found")
                    continue
                print(f"  -> Computing daily intensity for {var}")
                da_int = compute_intensity(da_event, var)
                if da_int is None:
                    continue
                save_intensity(var, da_int)

            print(f"  -> Aggregating {int_name} with agg_type='mean'")
            ds_int_FY = create_ds_of_aggregations_over_growing_season(
                da_int, int_name, CROP, "mean", grid,
                add_antecedent=False, antecedent_days=None,
            )

            # Set intensity to zero in season when there is no extreme event
            ds_int_FY = ds_int_FY.fillna(0)
            save_growing_season_output(int_name, ds_int_FY)


if __name__ == "__main__":
    main()
    