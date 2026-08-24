# #Code provided by DCCEEW
# #Regrids NARCliM2.0 from rotated pole to regular lon-lat using 
# 
# #This code regrids the Australia wide simulation at ~18km resolution
# #to 0.20 degrees (using this resolution as target as per correspondence with Jason Evans)
# #using conservative interpolation
# 
from pathlib import Path
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import xesmf
import pandas as pd
import os
import glob
import sys
import matplotlib.pyplot as plt

# 
# #Import regridding functions
# lib_path  = "/g/data/w97/amu561/CABLE_AWRA_comparison/scripts/Python/functions" # '/drought_metric_scripts/Drought_metrics/functions'
# 
# sys.path.append(os.path.abspath(lib_path))
# from narclim_regrid_functions import (
#     make_lat_lon_b,
#     create_target_grid,
#     run_regrid,
# )

#Output path (use scratch)
outpath = "/scratch/w97/mg5624/NARCLIM_regrid/"

#Input path
# in_path = "/g/data/zz63/NARCliM2-0/output-CMIP6/DD/AUS-18/NSW-Government" # change this to NARCliM2-0-SEAus-04 instead of AUS-18
in_path = "/g/data/zz63/NARCliM2-0/output-CMIP6/DD/NARCliM2-0-SEAus-04/NSW-Government" # change this to NARCliM2-0-SEAus-04 instead of AUS-18


#Variables to process
vars = [
    # "pr", 
    # "mrro", 
    # "mrso", 
    "tasmax",
    "tasmin"
]

#Experiments
#NARCliM also provides ssp-245 but runoff and soil moisture outputs are
#not currently available so excluding this scenario
experiments = [
    "historical", 
    "ssp126",
    "ssp245",
    "ssp370"
]

#List models
models = os.listdir(in_path)
# models = [
#     'ACCESS-ESM1-5'
# ]

### Set target grid ###

# Projections
plate_caree=ccrs.PlateCarree()
rot_pole=ccrs.RotatedPole(
    pole_longitude=141.38,
    pole_latitude=60.31,
    central_rotated_longitude=141.38
)

# Target grid details (interpolates to 20km grid across Australia)
# target_grid_details={
#     'lon':(
#         112, # 136, min longitude (corner of corner point)
#         155, # 158, max longitude (corner of corner point)
#         216, # 441, number of grid points + 1
#     ),
#     'lat':(
#         -45, # min latitude (corner of corner point)
#         -9,  # max latitude (corner of corner point)
#         181, # number of grid points + 1
#     )
# }

# Target grid details for 4km SEAus downscaled data
target_grid_details = {
    'lon': (
        140.0,   # min longitude (corner)
        154.0,   # max longitude (corner)
        351,     # number of grid points + 1
    ),
    'lat': (
        -40.0,   # min latitude (corner)
        -24.0,   # max latitude (corner)
        413,     # number of grid points + 1
    )
}

# Input file
# input_example_file=Path(str('/g/data/w97/amu561/CABLE_AWRA_comparison/test/' + 
# 'mrro_AUS-18_UKESM1-0-LL_historical_r1i1p1f2_NSW-Government_NARCliM2-0-WRF412R3_v1-r1_mon_195101-195112.nc'))
# 
# #'pr_AUS-18_ACCESS-ESM1-5_historical_r6i1p1f1_NSW-Government_NARCliM2-0-WRF412R3_v1-r1_mon_199501-199512.nc'))
# 
# run_path=Path().resolve()
# base_path=run_path.parent
# 
# # Output file
# output_base=run_path.joinpath('regridded')
# output_base.mkdir(exist_ok=True)
# 



# don't change this function if you don't know what it does...
# Used only to getnerate target grid from above dictionary
def generate_grid(grid_dict):
    lons=np.linspace(*grid_dict['lon'])
    lats=np.linspace(*grid_dict['lat'])
    return lons,lats

# don't change this function if you don't know what it does...
# Finds corners of rotated pole coordinates in rotated pole projection
def make_mid_points(x,dx):
    out=x-dx
    return np.append(out,dx+x[-1])

# don't change this function if you don't know what it does...
# Finds corners of rotated pole coordinates in plate-caree projection
def make_lat_lon_b(geo_lon,geo_lat):
    """a little more complicated than it would have needed
    to be if rlon was calculated correctly in the first place...
    """
    tmp = rot_pole.transform_points(plate_caree, geo_lon.values, geo_lat.values)
    rlon = tmp[:, :, 0]
    rlat = tmp[:, :, 1]

    rlon_1d=np.mean(rlon,axis=0)
    rlat_1d=np.mean(rlat,axis=1)
    dx=0.176
    rlonb_1d=make_mid_points(rlon_1d,0.5*dx)
    rlatb_1d=make_mid_points(rlat_1d,0.5*dx)

    rlonb=np.asarray(tuple(map(lambda x:rlonb_1d,rlatb_1d)))
    rlatb=np.asarray(tuple(map(lambda x:rlatb_1d,rlonb_1d))).transpose()

    tmp = plate_caree.transform_points(rot_pole, rlonb, rlatb)
    lon_b = tmp[:, :, 0]
    lat_b = tmp[:, :, 1]

    lon_b=xr.DataArray(lon_b,coords={'rlat_b':rlatb_1d,'rlon_b':rlonb_1d})
    lat_b=xr.DataArray(lat_b,coords={'rlat_b':rlatb_1d,'rlon_b':rlonb_1d})

    return lon_b, lat_b

# don't change this function if you don't know what it does...
# Uses generate grid to create target corner grids and uses that information
# to find mid points of grid
def create_target_grid(target_grid_details):
    tlon_b,tlat_b=generate_grid(target_grid_details)
    tlon=0.5*(tlon_b[:-1]+tlon_b[1:])
    tlat=0.5*(tlat_b[:-1]+tlat_b[1:])
    dummy=np.zeros((len(tlon),len(tlat)))
    target_grid=xr.DataArray(dummy,dims=['lon','lat']).to_dataset(name='target_grid')
    return target_grid.assign_coords({'lon':tlon,'lat':tlat,'lon_b':tlon_b,'lat_b':tlat_b}).transpose()

# don't change this function if you don't know what it does...
# Performs the regridding
def run_regrid(ds,target, var):

    if var == "tas":
        method = "bilinear"
    else:
        method = "conservative"

    regridder=xesmf.Regridder(
            ds,
            target,
            method, #'conservative',
            reuse_weights=False
        )

    regridded=regridder(ds,keep_attrs=True)
    return regridded.drop_vars(('lon_b','lat_b','rotated_pole','rlat_b','rlon_b'),errors='ignore')

#From NARCliM README.md
def open_mfdataset_fix(target_files,*args,**kwargs):
    def open_and_sanitize(nc_file):
        ds = xr.open_dataset(nc_file,*args,**kwargs)
        ds['crs'] = 0
        return ds
    ds = map(open_and_sanitize, target_files)
    ds = tuple(ds)
    return xr.combine_by_coords(
        ds,
        compat='no_conflicts',
        combine_attrs='drop_conflicts'
    )


# main function
def main(infiles, outfile, mask_file, var, target_grid):
    #ds=xr.open_mfdataset(infiles) # open netcdf file
    
    ds=open_mfdataset_fix(infiles)
    
    slon_b, slat_b=make_lat_lon_b(ds.lon,ds.lat)
    ds=ds.assign_coords({'lon_b':np.mod(slon_b,360),'lat_b':slat_b})
    ds['lon']=np.mod(ds.lon,360)
        
    #Can't run this to drop erroneous CRS dimension, leads to NA values
    #ds = ds.isel(crs=0, drop=True)

    ### Land masking ###
    
    #Runoff and soil moisture are already masked but need to mask 
    #precip and temperature
    #Won't do masking here, creates weird artifacts in the regridded data
    # if var == "tas" or var == "pr":
    # 
    #     #Read mask
    #     mask_ds = xr.open_dataset(mask_file)
    # 
    #     #Select variable
    #     mask = mask_ds["sftlf"]                     
    # 
    #     #Mask where land mask value is 0 (ocean)
    #     ds[var] = xr.where(mask == 0, np.nan, ds[var])

    
    
    regridded_ds=run_regrid(ds,target_grid, var)

    data = regridded_ds[var]  
    
    #For runoff and precip, multiple by the number of days per month and
    #seconds per day to go from mm/s to mm/day
    if var == "mrro" or var == "pr":
        data = data * 86400.0
        
    #Temperature from K to C
    if 'tas' in var:
        data = data - 273.15
                    
    
    ### Write output ###
    
    #Rename data variable
    data.name = var

    #Write to file
    data.to_netcdf(outfile, format='NETCDF4', 
                   encoding={var:{
                             'shuffle':True,
                             'chunksizes': [100, 100, 20], #[12, 20, 40],
                             'zlib':True,
                             'complevel':5}
                             })
    ds.close()
    # 
    # name=input_example_file.name
    # output_name=output_base.joinpath(name)
    # regridded_ds=regridded_ds.to_netcdf(output_name, format='NETCDF4', 
    #                                     encoding={vars[v]:{
    #                                     'shuffle':True,
    #                                     'chunksizes': [100, 100, 20], #[12, 20, 40],
    #                                     'zlib':True,
    #                                     'complevel':5}
    #                                     })

if __name__ == "__main__":
    #main()
    
    #Build target grid
    target_grid = create_target_grid(target_grid_details)

    #Loop through variables
    for v in range(len(vars)):
        
        print("Variable: " + vars[v])

        #Loop through models
        for m in range(len(models)):
            
            #Loop through experiments
            for e in range(len(experiments)):
                ensembles = os.listdir(str(in_path + "/" + models[m] + "/" + experiments[e]))

                #Loop through ensembles
                for ens in range(len(ensembles)):
                    ens_path = str(in_path + "/" + models[m] + "/" + experiments[e] +
                                   "/" + ensembles[ens])
                                   
                    #List RCMs
                    rcms = os.listdir(ens_path)
                    # rcms = ['NARCliM2-0-WRF412R3'] # only process this RCM for now
                    #Loop through RCMs
                    for r in range(len(rcms)):
                                            
                        #Final input data path
                        rcm_path = str(ens_path + "/" + rcms[r] + "/v1-r1/day/" + vars[v] + "/latest")
                        if not os.path.exists(rcm_path):
                            print(f"[SKIP] Path does not exist: {rcm_path}")
                            continue   # skip to next iteration of the loop
                        
                        #Progress
                        print(str("Processing: " + rcm_path))
                        
                        #Find input files
                        infiles = glob.glob(str(rcm_path + "/*.nc"))

                        #Stop if no files found
                        if len(infiles) == 0:
                            raise RuntimeError(str("No input files found: " + rcm_path))

                        #Find mask file
                        mask_file = glob.glob(str(ens_path + "/" + rcms[r] + 
                                                  "/v1-r1/fx/sftlf/latest/*.nc"))[0]

                        #Output path
                        outpath_final = str(outpath + "/" + experiments[e] + "/" + vars[v] +
                                            "/" + models[m] + "/" + rcms[r] + "/" + ensembles[ens])

                        #Create directory
                        if not os.path.exists(outpath_final):
                            os.makedirs(outpath_final)

                        #Output file name
                        outfile = Path(outpath_final) / (
                            experiments[e] + "_" + models[m] + "_" + rcms[r] + "_" +
                            ensembles[ens] + "_" + vars[v] + ".nc"
                        )

                        #If file already processed, continue
                        # if outfile.exists():
                        #     print("Skipping, already exists: " + rcm_path)
                        #     continue

                        #Run main function
                        main(infiles, outfile, mask_file, vars[v], target_grid)

    
    
    
    # 
    #     #Drop erroneous CRS dimension
    #     da = ds.pr.isel(time=0)
    #     min_da, max_da = dask.compute(da.min(skipna=True), da.max(skipna=True))
    # 
    #     print("min:", float(min_da.values))
    #     print("max:", float(max_da.values))
    # 
    # 
    # 
    #         #Drop erroneous CRS dimension
    #         import dask
    #         da1 = regridded_ds.pr.isel(time=0)
    #         min_da1, max_da1 = dask.compute(da1.min(skipna=True), da1.max(skipna=True))
    # 
    #         print("min:", float(min_da1.values))
    #         print("max:", float(max_da1.values))
    # 
    # 
    # 
    # import matplotlib.pyplot as plt
    # ds[vars[v]].isel(crs=0, time=0).plot()
    # plt.show()
    # 
    # 
    # 