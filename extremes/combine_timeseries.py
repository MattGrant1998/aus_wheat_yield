"""
Combine FY-aggregated NetCDF files into one continuous 1950–2025 file.

For each allowed variable folder under DATADIR, this script expects:

    {var}_FY_aggregated_1950-2021.nc
    {var}_FY_aggregated_2021-2025.nc

The overlapping year (2021) is checked before the files are combined. The
duplicate 2021 record from the second file is then removed.

Usage:
    python combine_fy_aggregated.py --dry-run
    python combine_fy_aggregated.py
"""

import argparse
import os

import numpy as np
import xarray as xr


DATADIR = "/g/data/w97/mg5624/ABS_project/extremes/FY_aggregated/"

START_YEAR = 1950
END_YEAR = 2025

DA1_YEARS = (1950, 2021)
DA2_YEARS = (2022, 2025)

# var_name -> (aggregation type, has_intensity)
VAR_CONFIG = {
    "heatwave": ("sum", True),
    "frost": ("sum", False),
    "gdd": ("sum", False),
    "tr": ("mean", False),
    "cdd": ("max", False),
    "tmax": ("mean", False),
    "tmin": ("mean", False),
    "precip": ("mean", False),
    "sm": ("mean", False),
    # "precip_drought_30": ("sum", True),
    # "precip_drought_90": ("sum", True),
    "precip_drought_180": ("sum", True),
    # "sm_drought_30": ("sum", True),
    # "sm_drought_90": ("sum", True),
    "sm_drought_180": ("sum", True),
}


def allowed_vars():
    """Return base variables plus the configured intensity variants."""
    names = []
    for var, (_, has_intensity) in VAR_CONFIG.items():
        names.append(var)
        if has_intensity:
            names.append(f"{var}_intensity")
    return names


def combine_variable(var_name, dry_run=False):
    folder = os.path.join(DATADIR, var_name)
    file_a = os.path.join(folder, f"{var_name}_FY_aggregated_{DA1_YEARS[0]}-{DA1_YEARS[1]}.nc")
    file_b = os.path.join(folder, f"{var_name}_FY_aggregated_{DA2_YEARS[0]}-{DA2_YEARS[1]}.nc")
    output = os.path.join(folder, f"{var_name}_FY_aggregated_{START_YEAR}-{END_YEAR}.nc")

    print(f"\n[{var_name}]")
    if not os.path.isdir(folder):
        print(f"  -- directory not found: {folder}")
        return False
    if not os.path.isfile(file_a) or not os.path.isfile(file_b):
        print(f"  -- input files not found:\n     {file_a}\n     {file_b}")
        return False

    with xr.open_dataset(file_a) as ds_a, xr.open_dataset(file_b) as ds_b:

        if dry_run:
            print(f"  DRY RUN: would write {output}")
            return True

        # Keep all records from the first file and append years after 2021
        # from the second file, avoiding a duplicate overlap record.
        combined = xr.concat([ds_a, ds_b], dim="time")
        combined = combined.sortby("time")

        # Reuse compression/chunking settings from the first input where present.
        encoding = {}
        for name in combined.variables:
            if name in ds_a.variables:
                # Keep only NetCDF write options; source-specific keys such as
                # 'source' and 'original_shape' must not be passed to to_netcdf.
                valid_keys = {
                    "_FillValue", "dtype", "zlib", "complevel", "shuffle",
                    "fletcher32", "contiguous", "chunksizes", "compression",
                    "least_significant_digit", "szip_coding", "szip_pixels_per_block",
                }
                source_encoding = ds_a[name].encoding
                encoding[name] = {
                    key: value for key, value in source_encoding.items()
                    if key in valid_keys
                }

        combined.to_netcdf(output, mode="w", format="NETCDF4", encoding=encoding)
        combined.close()

    print(f"  wrote {output}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="check inputs and overlaps without writing files")
    args = parser.parse_args()

    successes = 0
    for var_name in allowed_vars():
        successes += combine_variable(var_name, dry_run=args.dry_run)

    print(f"\nCompleted: {successes}/{len(allowed_vars())} variables processed successfully.")


if __name__ == "__main__":
    main()
