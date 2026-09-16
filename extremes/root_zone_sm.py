import xarray as xr
# import processing_functions

def find_root_zone_soil_moisture(year):
    """
    Finds the root zone soil moisture by adding s0 and ss together. 
    """
    awra_path = '/g/data/iu04/australian-water-outlook/historical/v1/AWRALv7/processed/values/day/'
    sm = xr.open_dataset(f'{awra_path}sm_{year}.nc')['sm']
    # ss = xr.open_dataset(f'{awra_path}ss_pct_{year}.nc')['ss']
    # soil_moisture_root = s0 + ss

    sm.name = 'awra_root_zone_soil_moisture'
    return sm


# def process_root_soil_moisture():
#     """
#     Applies the processing functions to root zone soil moisture to ensure same format as rest of data
#     """
#     SM_root = find_root_zone_soil_moisture()
#     SM_root_processed = \
#     processing_functions.set_time_coord_to_year_month_01_datetime(
#         processing_functions.regrid_to_AGCD_grid(
#             processing_functions.constrain_to_australia(
#                 processing_functions.rename_coord_titles_to_lat_long(
#                     SM_root
#                 )
#             )
#         )
#     )

#     return SM_root_processed

def save_SM_root(year):
    SM_root = find_root_zone_soil_moisture(year)
    print(f'SM at {year} processed')
    SM_root.to_netcdf(f'/g/data/w97/mg5624/RF_project/AWRA_SM/daily/root_zone_soil_moisture_{year}.nc')
    print(f'SM at {year} saved')


def main():
    years = list(range(2021, 2026))
    for year in years:
        save_SM_root(year)


if __name__ == "__main__":
    main()