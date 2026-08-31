import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# =====================================================================
# CONFIG
# =====================================================================

AGCD_ROOT = Path("/g/data/zv2/agcd/v1-0-4")
AWRA_ROOT = Path("/g/data/w97/mg5624/RF_project/AWRA_SM/daily")
OUT_ROOT = Path("/g/data/w97/mg5624/ABS_project/extremes/daily")

# AGCD aggregation label used in its filenames/paths
AGCD_AGG = {
    "tmax": "mean",
    "tmin": "mean",
    "precip": "total",
}

# Name of the data variable *inside* each NetCDF file. AGCD/AWRA-L
# internal variable names don't always match the path/filename — check
# with e.g. `xr.open_dataset(path).data_vars` and fix here if needed.
NC_VAR_NAME = {
    "tmax": "tmax",
    "tmin": "tmin",
    "precip": "precip",
    "sm": "root_zone_soil_moisture",
}

# Full period to process and the baseline used for percentile thresholds
START_YEAR = 2021
END_YEAR = 2025
BASELINE = (1990, 2021)

# drought settings (unchanged from original)
DROUGHT_WINDOWS = [30, 90, 180]
DROUGHT_THRESHOLD = 0.15
DROUGHT_THRESHOLD_TYPE = "percentile"


# =====================================================================
# Generic index-calculation logic (unchanged from NARCLIM version)
# =====================================================================

def rolling_data(data, n_days, agg):
    if agg == "sum":
        out = data.rolling(time=n_days, center=False).sum()
    elif agg == "mean":
        out = data.rolling(time=n_days, center=False).mean()
    else:
        raise ValueError("agg must be 'sum' or 'mean'")
    out = out.rename(f"{data.name}_rolling_{n_days}_{agg}")
    return out


def ensure_datetime64(ds):
    """Force time coordinate to datetime64[ns] if it is object/cftime dtype."""
    if ds.indexes["time"].dtype == "O":
        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))
    return ds


def find_percentile_at_grid(rolling_data_da, baseline, threshold):
    # ---------------------------------------------------------
    # Select baseline period
    # ---------------------------------------------------------
    if baseline == "full_ts":
        baseline_data = rolling_data_da
    else:
        start_year, end_year = baseline
        baseline_data = rolling_data_da.sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )
    baseline_data = baseline_data.chunk({"time": -1, "lat": -1, "lon": -1})

    # ---------------------------------------------------------
    # Detect calendar type from actual data, not metadata
    # ---------------------------------------------------------
    md = set(zip(baseline_data["time.month"].values,
                 baseline_data["time.day"].values))

    has_feb29 = (2, 29) in md
    has_feb30 = (2, 30) in md
    has_apr31 = (4, 31) in md
    has_jun31 = (6, 31) in md

    should_insert_feb29 = has_feb29 and not (has_feb30 or has_apr31 or has_jun31)

    if not should_insert_feb29:
        if threshold == "mean":
            percentiles = baseline_data.groupby("time.dayofyear").mean(dim="time")
            var_name = percentiles.name
            percentiles = percentiles.rename(f"{var_name}_mean")
        else:
            percentiles = baseline_data.groupby("time.dayofyear").quantile(
                threshold, dim="time"
            )
        return percentiles

    # ---------------------------------------------------------
    # Feb 29 insertion logic (real leap-year / Gregorian calendars)
    # ---------------------------------------------------------
    years = np.unique(baseline_data["time.year"].values)

    non_leap_years = [
        y for y in years
        if (2, 29) not in set(zip(
            baseline_data.sel(time=str(y))["time.month"].values,
            baseline_data.sel(time=str(y))["time.day"].values
        ))
    ]

    feb28 = baseline_data.sel(
        time=(
            (baseline_data["time.month"] == 2)
            & (baseline_data["time.day"] == 28)
            & baseline_data["time.year"].isin(non_leap_years)
        )
    )

    feb28_coords = feb28["time"].values + np.timedelta64(1, "D")
    feb28 = feb28.assign_coords(time=("time", feb28_coords))

    baseline_with_feb29 = xr.concat([baseline_data, feb28], dim="time").sortby("time")
    baseline_with_feb29 = baseline_with_feb29.chunk({"time": -1, "lat": -1, "lon": -1})

    if threshold == "mean":
        percentiles = baseline_with_feb29.groupby("time.dayofyear").mean(dim="time")
        var_name = percentiles.name
        percentiles = percentiles.rename(f"{var_name}_mean")
    else:
        percentiles = baseline_with_feb29.groupby("time.dayofyear").quantile(
            threshold, dim="time"
        )

    return percentiles


def find_drought_days(hydrological_data, var_name, n_days, threshold,
                      threshold_type="percentile", baseline="full_ts"):
    print('Raw data: \n', hydrological_data)
    hyd_rolling = rolling_data(hydrological_data, n_days, "mean")

    if threshold_type == "percentile":
        print('Rolling data: \n', hyd_rolling)
        percentile_data = find_percentile_at_grid(hyd_rolling, baseline, threshold)
        print('percentile data: \n', percentile_data)
        thresholds = percentile_data.sel(dayofyear=hyd_rolling["time.dayofyear"])
        drought_days = hyd_rolling.where(hyd_rolling < thresholds)
    elif threshold_type == "absolute":
        drought_days = hyd_rolling.where(hyd_rolling < threshold)
    else:
        raise ValueError("threshold_type must be 'percentile' or 'absolute'")

    drought_days = drought_days.rename(f"{n_days}_day_{var_name}_drought")
    return drought_days


def find_heatwave_days(tmax_daily, threshold=32.0):
    hw = tmax_daily.where(tmax_daily > threshold)
    hw = hw.rename("heatwave")
    return hw


def find_frost_days(tmin_daily):
    frost = tmin_daily.where(tmin_daily < 0)
    frost = frost.rename("frost")
    return frost


def growing_degree_days(tmax, tmin, basetemp=0.0):
    tavg = (tmax + tmin) / 2.0
    gdd = tavg - basetemp
    gdd = gdd.where(gdd > 0, 0)
    gdd = gdd.rename("gdd")
    return gdd


def temperature_range(tmax, tmin):
    tr = tmax - tmin
    tr = tr.rename("tr")
    return tr


def _consecutive_dry_np(is_dry):
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
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )
    consecutive = consecutive.rename("cdd")
    return consecutive


# =====================================================================
# AGCD / AWRA-L specific file discovery + opening
# =====================================================================

def list_agcd_files(variable, start_year, end_year):
    agg = AGCD_AGG[variable]
    files = []
    for year in range(start_year, end_year + 1):
        f = (AGCD_ROOT / variable / agg / "r005" / "01day"
             / f"agcd_v1_{variable}_{agg}_r005_daily_{year}.nc")
        if f.exists():
            files.append(f)
        else:
            print(f"[WARN] missing AGCD file: {f}")
    return sorted(files)


def open_agcd_var(variable, start_year, end_year):
    """Open+concat AGCD daily files for one variable across years."""
    files = list_agcd_files(variable, start_year, end_year)
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ensure_datetime64(ds)
    return ds


def list_awra_files(start_year, end_year):
    files = []
    for year in range(start_year, end_year + 1):
        f = AWRA_ROOT / f"root_zone_soil_moisture_{year}.nc"
        if f.exists():
            files.append(f)
        else:
            print(f"[WARN] missing AWRA-L file: {f}")
    return sorted(files)


def open_awra_sm(start_year, end_year):
    """Open+concat AWRA-L root-zone soil moisture files across years."""
    files = list_awra_files(start_year, end_year)
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ensure_datetime64(ds)
    return ds


def save_da(out_root, var_name, da):
    """Save a DataArray to out_root/var_name/var_name.nc"""
    out_dir = out_root / var_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{var_name}.nc"
    da.to_netcdf(out_path)
    print(f"Saved {var_name} -> {out_path}")
    return out_path


# =====================================================================
# Main workflow
# =====================================================================

def main():
    print(f"=== AGCD / AWRA-L observations: {START_YEAR}-{END_YEAR} ===")

    ds_tmax = open_agcd_var("tmax", START_YEAR, END_YEAR)
    ds_tmin = open_agcd_var("tmin", START_YEAR, END_YEAR)
    ds_precip = open_agcd_var("precip", START_YEAR, END_YEAR)
    ds_sm = open_awra_sm(START_YEAR, END_YEAR)

    # ------------------------------------------------------------------
    # heatwave (tmax > 32)
    # ------------------------------------------------------------------
    if ds_tmax is not None:
        out_path = OUT_ROOT / "heatwave" / "heatwave.nc"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
        else:
            hw = find_heatwave_days(ds_tmax[NC_VAR_NAME["tmax"]], threshold=32.0)
            save_da(OUT_ROOT, "heatwave", hw)

    # ------------------------------------------------------------------
    # frost (tmin < 0)
    # ------------------------------------------------------------------
    if ds_tmin is not None:
        out_path = OUT_ROOT / "frost" / "frost.nc"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
        else:
            frost = find_frost_days(ds_tmin[NC_VAR_NAME["tmin"]])
            save_da(OUT_ROOT, "frost", frost)

    # ------------------------------------------------------------------
    # gdd (tmax, tmin)
    # ------------------------------------------------------------------
    if ds_tmax is not None and ds_tmin is not None:
        out_path = OUT_ROOT / "gdd" / "gdd.nc"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
        else:
            gdd = growing_degree_days(
                ds_tmax[NC_VAR_NAME["tmax"]], ds_tmin[NC_VAR_NAME["tmin"]], basetemp=0.0
            )
            save_da(OUT_ROOT, "gdd", gdd)

    # ------------------------------------------------------------------
    # tr (tmax, tmin)
    # ------------------------------------------------------------------
    if ds_tmax is not None and ds_tmin is not None:
        out_path = OUT_ROOT / "tr" / "tr.nc"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
        else:
            tr = temperature_range(ds_tmax[NC_VAR_NAME["tmax"]], ds_tmin[NC_VAR_NAME["tmin"]])
            save_da(OUT_ROOT, "tr", tr)

    # ------------------------------------------------------------------
    # cdd (precip)
    # ------------------------------------------------------------------
    if ds_precip is not None:
        out_path = OUT_ROOT / "cdd" / "cdd.nc"
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
        else:
            cdd = consecutive_dry_days(ds_precip[NC_VAR_NAME["precip"]])
            save_da(OUT_ROOT, "cdd", cdd)

    # ------------------------------------------------------------------
    # drought (precip, N-day rolling, percentile threshold)
    # ------------------------------------------------------------------
    if ds_precip is not None:
        for w in DROUGHT_WINDOWS:
            var_name = f"precip_drought_{w}"
            out_path = OUT_ROOT / var_name / f"{var_name}.nc"
            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue
            precip_drought = find_drought_days(
                ds_precip[NC_VAR_NAME["precip"]],
                "precip",
                w,
                DROUGHT_THRESHOLD,
                threshold_type=DROUGHT_THRESHOLD_TYPE,
                baseline=BASELINE,
            )
            precip_drought = precip_drought.rename(var_name)
            save_da(OUT_ROOT, var_name, precip_drought)

    # ------------------------------------------------------------------
    # drought (soil moisture, N-day rolling, percentile threshold)
    # ------------------------------------------------------------------
    if ds_sm is not None:
        for w in DROUGHT_WINDOWS:
            var_name = f"sm_drought_{w}"
            out_path = OUT_ROOT / var_name / f"{var_name}.nc"
            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue
            sm_drought = find_drought_days(
                ds_sm[NC_VAR_NAME["sm"]],
                "sm",
                w,
                DROUGHT_THRESHOLD,
                threshold_type=DROUGHT_THRESHOLD_TYPE,
                baseline=BASELINE,
            )
            sm_drought = sm_drought.rename(var_name)
            save_da(OUT_ROOT, var_name, sm_drought)


if __name__ == "__main__":
    main()
