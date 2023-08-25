import argparse
import datetime
import glob
import logging
import os
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout, )


def create_inflow_file_for_all_inputs(lsm_files: str or list,
                                      inputs_root_dir: str,
                                      inflows_root_dir: str,
                                      weight_table: str,
                                      comid_lat_lon_z: str = 'comid_lat_lon_z.csv') -> None:
    """
    Generate inflow files for use with RAPID. The generated inflow file will sort the river ids in the order found in
    the comid_lat_lon_z csv.

    Parameters
    ----------
    lsm_files: str or list
        Path to the LSM files or list of LSM files or a glob pattern string
    inputs_root_dir: str
        Path to the root vpudir for input directories
    inflows_root_dir: str
        Path to the root vpudir for inflow directories
    weight_table: str, list
        name of the weight table to use from each input vpudir
    comid_lat_lon_z: str
        name of the comid_lat_lon_z csv file to use from each vpudir
    """
    # open all the ncs and select only the area within the weight table
    logging.info('Opening LSM multifile dataset')
    lsm_dataset = xr.open_mfdataset(lsm_files)

    # Select the variable names
    runoff_variable = [x for x in ['ro', 'RO', 'runoff', 'RUNOFF'] if x in lsm_dataset.variables][0]
    lon_variable = [x for x in ['lon', 'longitude', 'LONGITUDE', 'LON'] if x in lsm_dataset.variables][0]
    lat_variable = [x for x in ['lat', 'latitude', 'LATITUDE', 'LAT'] if x in lsm_dataset.variables][0]

    # Check that the weight table dimensions match the dataset dimensions
    if f'{lsm_dataset[lat_variable].shape[0]}x{lsm_dataset[lon_variable].shape[0]}' not in weight_table:
        raise ValueError(f"Weight table dimensions don't match the input dataset dimensions: {weight_table}")

    # Get conversion factor
    logging.info('Getting conversion factor')
    conversion_factor = 1
    units = lsm_dataset[runoff_variable].attrs.get('units', False)
    if not units:
        logging.warning("No units attribute found. Assuming meters")
    elif units == 'm':
        conversion_factor = 1
    elif units == 'mm':
        conversion_factor = .001
    else:
        raise ValueError(f"Unknown units: {units}")

    # get the time array from the dataset
    logging.info('Reading Time Values')
    datetime_array = lsm_dataset['time'].values.flatten()

    for input_dir in sorted([d for d in glob.glob(os.path.join(inputs_root_dir, '*')) if os.path.isdir(d)]):
        logging.info(f'Processing {input_dir}')
        vpu_name = os.path.basename(input_dir)
        wt_path = os.path.join(input_dir, weight_table)
        comid_path = os.path.join(input_dir, comid_lat_lon_z)
        os.makedirs(os.path.join(inflows_root_dir, vpu_name), exist_ok=True)
        start_date = datetime.datetime.utcfromtimestamp(datetime_array[0].astype(float) / 1e9).strftime('%Y%m%d')
        end_date = datetime.datetime.utcfromtimestamp(datetime_array[-1].astype(float) / 1e9).strftime('%Y%m%d')
        inflow_file_path = os.path.join(inflows_root_dir, vpu_name, f'm3_{vpu_name}_{start_date}_{end_date}.nc')

        if os.path.exists(inflow_file_path):
            continue

        # Ensure that every input file exists
        if not os.path.exists(wt_path):
            raise FileNotFoundError(f'{wt_path} does not exist')
        if not os.path.exists(comid_path):
            raise FileNotFoundError(f'{comid_path} does not exist')

        # load in weight table and get some information
        logging.info('\tReading weight table and comid_lat_lon_z csvs')
        weight_df = pd.read_csv(wt_path)
        comid_df = pd.read_csv(comid_path)

        logging.info('\tCreating inflow array')
        if lsm_dataset[runoff_variable].ndim == 3:
            inflow_df = lsm_dataset[runoff_variable].values[:, weight_df['lat_index'].values,
                        weight_df['lon_index'].values]
        elif lsm_dataset[runoff_variable].ndim == 4:
            inflow_df = lsm_dataset[runoff_variable].values[:, :, weight_df['lat_index'].values,
                        weight_df['lon_index'].values]
            inflow_df = np.where(np.isnan(inflow_df[:, 0, :]), inflow_df[:, 1, :], inflow_df[:, 0, :]),
        else:
            raise ValueError(f"Unknown number of dimensions: {lsm_dataset.ndim}")
        # correct nans, negatives, and units
        inflow_df = np.nan_to_num(inflow_df, nan=0)
        inflow_df[inflow_df < 0] = 0
        inflow_df = inflow_df * weight_df['area_sqm'].values * conversion_factor
        # group columns by matching weight table rivid
        inflow_df = pd.DataFrame(inflow_df, columns=weight_df.iloc[:, 0].to_numpy(), index=datetime_array)
        inflow_df = inflow_df.groupby(by=weight_df.iloc[:, 0].to_numpy(), axis=1).sum()
        inflow_df = inflow_df[comid_df.iloc[:, 0].to_numpy()]

        # Create output inflow netcdf data
        logging.info("\tWriting inflows to file")

        try:
            with nc.Dataset(inflow_file_path, "w", format="NETCDF3_CLASSIC") as ds:
                # create dimensions
                ds.createDimension('time', inflow_df.shape[0])
                ds.createDimension('rivid', comid_df.shape[0])
                ds.createDimension('nv', 2)

                ds.createVariable('rivid', 'i4', ('rivid',))
                ds.createVariable('time', 'i4', ('time',))
                ds.createVariable('time_bnds', 'i4', ('time', 'nv',))
                ds.createVariable('m3_riv', 'f4', ('time', 'rivid'), fill_value=0.0)
                ds.createVariable('lon', 'f8', ('rivid',), fill_value=-9999.0)
                ds.createVariable('lat', 'f8', ('rivid',), fill_value=-9999.0)
                ds.createVariable('crs', 'i4')

                # rivid
                ds['rivid'].long_name = 'unique identifier for each river reach'
                ds['rivid'].units = '1'
                ds['rivid'].cf_role = 'timeseries_id'
                ds['rivid'][:] = comid_df.iloc[:, 0].astype(int).to_numpy()

                # time
                reference_time = datetime_array[0]
                time_step = (datetime_array[1] - datetime_array[0]).astype('timedelta64[s]')
                ds['time'].long_name = 'time'
                ds['time'].standard_name = 'time'
                ds['time'].units = f'seconds since {reference_time.astype("datetime64[s]")}'  # Must be seconds
                ds['time'].axis = 'T'
                ds['time'].calendar = 'gregorian'
                ds['time'].bounds = 'time_bnds'
                ds['time'][:] = (datetime_array - reference_time).astype('timedelta64[s]').astype(int)

                # time_bnds
                time_bnds_array = np.stack([datetime_array, datetime_array + time_step], axis=1)
                time_bnds_array = (time_bnds_array - reference_time).astype('timedelta64[s]').astype(int)
                ds['time_bnds'][:] = time_bnds_array

                # m3_riv
                ds['m3_riv'][:] = inflow_df.values
                ds['m3_riv'].long_name = 'accumulated inflow volume in river reach boundaries'
                ds['m3_riv'].units = 'm3'
                ds['m3_riv'].coordinates = 'lon lat'
                ds['m3_riv'].grid_mapping = 'crs'
                ds['m3_riv'].cell_methods = "time: sum"

                # longitude
                ds['lon'].long_name = 'longitude of a point related to each river reach'
                ds['lon'].standard_name = 'longitude'
                ds['lon'].units = 'degrees_east'
                ds['lon'].axis = 'X'
                ds['lon'][:] = comid_df['lon'].values

                # latitude
                ds['lat'].long_name = 'latitude of a point related to each river reach'
                ds['lat'].standard_name = 'latitude'
                ds['lat'].units = 'degrees_north'
                ds['lat'].axis = 'Y'
                ds['lat'][:] = comid_df['lat'].values

                # crs
                ds['crs'].grid_mapping_name = 'latitude_longitude'
                ds['crs'].epsg_code = 'EPSG:4326'  # WGS 84
                ds['crs'].semi_major_axis = 6378137.0
                ds['crs'].inverse_flattening = 298.257223563

                # add global attributes
                ds.Conventions = 'CF-1.6'
                ds.history = 'date_created: {0}'.format(datetime.datetime.utcnow())
                ds.featureType = 'timeSeries'
        except Exception as e:
            logging.error(f"\tError writing inflow file: {e}")
            if os.path.exists(inflow_file_path):
                os.remove(inflow_file_path)
            continue

    lsm_dataset.close()
    return


def main():
    parser = argparse.ArgumentParser(description='Create inflow file for LSM files and input vpudir.')

    # Define the command-line argument
    parser.add_argument('--lsmfile', type=str, help='LSM file containing RO variable')
    parser.add_argument('--inputsroot', type=str, help='Inputs vpudir')
    parser.add_argument('--inflowsroot', type=str, help='Inflows vpudir')
    args = parser.parse_args()

    # Access the parsed argument
    if not all([args.lsmfile, args.inputsroot, args.inflowsroot]):
        raise ValueError('Missing required arguments')

    # Create the inflow file for each input vpudir with the given LSM file
    create_inflow_file_for_all_inputs(args.lsmfile,
                                      args.inputsroot,
                                      args.inflowsroot,
                                      weight_table='weight_era5_721x1440.csv',
                                      comid_lat_lon_z='comid_lat_lon_z.csv')


if __name__ == '__main__':
    main()
