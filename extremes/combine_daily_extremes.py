import xarray as xr

def main():
    vars = [
        'cdd',
        # '30_met_drought_centered',
        'tr', 
        'gdd'
    ]
    for var in vars:
        print(f'Combining: {var}')
        print('------------------')
        var_name = f'{var}_1950-2021'
        # path = '/scratch/w97/mg5624/extremes/daily/'
        path = '/g/data/w97/mg5624/ABS_project/extremes/daily/'
        ds1 = xr.open_dataset(path + f'{var_name}_1.nc')
        ds2 = xr.open_dataset(path + f'{var_name}_2.nc')
        ds2 = ds2.sel(lon=ds2.lon < 135)
        var_name_out = f'{var}_1950-2021'
        combined_ds = xr.combine_by_coords([ds1, ds2], combine_attrs='override')
        combined_ds.to_netcdf(path + f'{var_name_out}.nc', mode='w')
        ds1.close()
        ds2.close()
        combined_ds.close()


if __name__ == '__main__':
    main()