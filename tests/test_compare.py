"""
We are testing the first 10 days (1980-1-1 to 1980-1-10) and the last 10 rivers in the order of the comid_lat_lon_z. The input data is that subselection
Things to test:
    - dimensions match
    - rivid order matches
    - m3 values match
    - time matches
    - time bnds match
    - lon match
    - lat match
    - crs is EPSG 4326
"""
import glob
import netCDF4 as nc
import sys
import os

# Add the project_root vpudir to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from inflow_fast import create_inflow_file

def check_function(validation_ds, output_ds, test):
    print(test)
    try:
        # Check dimensions match
        assert output_ds.dimensions.keys() == validation_ds.dimensions.keys(), "Dimensions do not match."

        for key in output_ds.dimensions.keys():
            if key == 'nv':
                continue
            assert (output_ds[key][:] == validation_ds[key][:]).all(), f"{key} values differ"
        
        # Check m3 values match
        assert (output_ds['m3_riv'][:] == validation_ds['m3_riv'][:]).all(), "m3 values do not match."
        
        # Check time bounds match
        assert (output_ds['time_bnds'][:] == validation_ds['time_bnds'][:]).all(), "time bounds do not match."
        
        # Check lon match
        assert (output_ds['lon'][:] == validation_ds['lon'][:]).all(), "lon values do not match."
        
        # Check lat match
        assert (output_ds['lat'][:] == validation_ds['lat'][:]).all(), "lat values do not match."
        
        # Check CRS is EPSG 4326
        assert output_ds['crs'].epsg_code == validation_ds['crs'].epsg_code, f"CRS is not EPSG 4326. CRS is {output_ds['crs'].epsg_code}"
        
        print("All tests passed.")
        
    except AssertionError as e:
        print(f"Test failed: {e}")

    finally:
        # Close the datasets
        output_ds.close()
        validation_ds.close()

# TEST 1: Normal inputs
create_inflow_file(glob.glob('./tests/inputs/era5_721x1440_sample_data/*.nc'),
                   './tests/inputs/weight_era5_721x1440_last_10.csv', './tests/inputs/comid_lat_lon_z_last_10.csv',
                   './tests/test.nc')

out_ds = nc.Dataset('./tests/test.nc', 'r')
val_ds = nc.Dataset('tests/validation/1980_01_01to10_last10.nc', 'r')

check_function(val_ds, out_ds, 'TEST 1: Normal inputs')