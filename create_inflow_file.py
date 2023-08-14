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
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        Path to the root directory for input directories
    inflows_root_dir: str
        Path to the root directory for inflow directories
    weight_table: str, list
        name of the weight table to use from each input directory
    comid_lat_lon_z: str
        name of the comid_lat_lon_z csv file to use from each directory
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

    for input_dir in [d for d in glob.glob(os.path.join(inputs_root_dir, '*')) if os.path.isdir(d)]:
        logging.info(f'Processing {input_dir}')
        vpu_name = os.path.basename(input_dir)
        wt_path = os.path.join(input_dir, weight_table)
        comid_path = os.path.join(input_dir, comid_lat_lon_z)

        # Ensure that every input file exists
        if not os.path.exists(wt_path):
            raise FileNotFoundError(f'{wt_path} does not exist')
        if not os.path.exists(comid_path):
            raise FileNotFoundError(f'{comid_path} does not exist')

        # load in weight table and get some information
        logging.info('\tReading weight table and comid_lat_lon_z csvs')
        weight_df = pd.read_csv(wt_path)
        comid_df = pd.read_csv(comid_path)

        min_lon_idx = weight_df['lon_index'].min().astype(int)
        max_lon_idx = weight_df['lon_index'].max().astype(int)
        min_lat_idx = weight_df['lat_index'].min().astype(int)
        max_lat_idx = weight_df['lat_index'].max().astype(int)

        if min_lon_idx > max_lon_idx:
            min_lon_idx, max_lon_idx = max_lon_idx, min_lon_idx
        if min_lat_idx > max_lat_idx:
            min_lat_idx, max_lat_idx = max_lat_idx, min_lat_idx

        # for readability, select certain cols from the weight table
        lat_indices = weight_df['lat_index'].values - min_lat_idx
        lon_indices = weight_df['lon_index'].values - min_lon_idx

        logging.info('\tCreating inflow array')
        if lsm_dataset[runoff_variable].ndim == 3:
            inflow_df = lsm_dataset[runoff_variable].values[:, lat_indices, lon_indices]
        elif lsm_dataset[runoff_variable].ndim == 4:
            inflow_df = lsm_dataset[runoff_variable].values[:, :, lat_indices, lon_indices]
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
        # group dataframe by year month day values in index
        # inflow_df = inflow_df.groupby(inflow_df.index.strftime('%Y%m%d')).sum()
        # inflow_df.index = pd.to_datetime(inflow_df.index, format='%Y%m%d')
        # datetime_array = inflow_df.index.to_numpy()
        # get an array from the dataframe sorted by the rivid order in the comid_lat_lon_z csv
        inflow_df = inflow_df[comid_df.iloc[:, 0].to_numpy()]

        # Create output inflow netcdf data
        logging.info("\tWriting inflows to file")
        os.makedirs(os.path.join(inflows_root_dir, vpu_name), exist_ok=True)
        start_date = datetime.datetime.utcfromtimestamp(datetime_array[0].astype(float) / 1e9).strftime('%Y%m%d')
        end_date = datetime.datetime.utcfromtimestamp(datetime_array[-1].astype(float) / 1e9).strftime('%Y%m%d')
        inflow_file_path = os.path.join(inflows_root_dir, vpu_name, f'm3_{vpu_name}_{start_date}_{end_date}.nc')

        with nc.Dataset(inflow_file_path, "w", format="NETCDF3_CLASSIC") as inflow_nc:
            # create dimensions
            inflow_nc.createDimension('time', inflow_df.shape[0])
            inflow_nc.createDimension('rivid', comid_df.shape[0])
            inflow_nc.createDimension('nv', 2)

            # m3_riv
            m3_riv_var = inflow_nc.createVariable('m3_riv', 'f4', ('time', 'rivid'),
                                                  fill_value=0, zlib=True, complevel=7)
            m3_riv_var[:] = inflow_df.to_numpy()
            m3_riv_var.long_name = 'accumulated inflow inflow volume in river reach boundaries'
            m3_riv_var.units = 'm3'
            m3_riv_var.coordinates = 'lon lat'
            m3_riv_var.grid_mapping = 'crs'
            m3_riv_var.cell_methods = "time: sum"

            # rivid
            rivid_var = inflow_nc.createVariable('rivid', 'i4', ('rivid',),
                                                 zlib=True, complevel=7)
            rivid_var[:] = comid_df.iloc[:, 0].to_numpy()
            rivid_var.long_name = 'unique identifier for each river reach'
            rivid_var.units = '1'
            rivid_var.cf_role = 'timeseries_id'

            # time
            reference_time = datetime_array[0]
            time_step = (datetime_array[1] - datetime_array[0]).astype('timedelta64[s]')
            time_var = inflow_nc.createVariable('time', 'i4', ('time',),
                                                zlib=True, complevel=7)
            time_var[:] = (datetime_array - reference_time).astype('timedelta64[s]').astype(int)
            time_var.long_name = 'time'
            time_var.standard_name = 'time'
            time_var.units = f'seconds since {reference_time.astype("datetime64[s]")}'  # Must be seconds
            time_var.axis = 'T'
            time_var.calendar = 'gregorian'
            time_var.bounds = 'time_bnds'

            # time_bnds
            time_bnds = inflow_nc.createVariable('time_bnds', 'i4', ('time', 'nv',),
                                                 zlib=True, complevel=7)
            time_bnds_array = np.stack([datetime_array, datetime_array + time_step], axis=1)
            time_bnds_array = (time_bnds_array - reference_time).astype('timedelta64[s]').astype(int)
            time_bnds[:] = time_bnds_array

            # longitude
            lon_var = inflow_nc.createVariable('lon', 'f8', ('rivid',),
                                               fill_value=-9999.0, zlib=True, complevel=7)
            lon_var[:] = comid_df['lon'].values
            lon_var.long_name = 'longitude of a point related to each river reach'
            lon_var.standard_name = 'longitude'
            lon_var.units = 'degrees_east'
            lon_var.axis = 'X'

            # latitude
            lat_var = inflow_nc.createVariable('lat', 'f8', ('rivid',),
                                               fill_value=-9999.0, zlib=True, complevel=7)
            lat_var[:] = comid_df['lat'].values
            lat_var.long_name = 'latitude of a point related to each river reach'
            lat_var.standard_name = 'latitude'
            lat_var.units = 'degrees_north'
            lat_var.axis = 'Y'

            # crs
            crs_var = inflow_nc.createVariable('crs', 'i4',
                                               zlib=True, complevel=7)
            crs_var.grid_mapping_name = 'latitude_longitude'
            crs_var.epsg_code = 'EPSG:4326'  # WGS 84
            crs_var.semi_major_axis = 6378137.0
            crs_var.inverse_flattening = 298.257223563

            # add global attributes
            inflow_nc.Conventions = 'CF-1.6'
            inflow_nc.history = 'date_created: {0}'.format(datetime.datetime.utcnow())
            inflow_nc.featureType = 'timeSeries'

    lsm_dataset.close()
    return


def main():
    parser = argparse.ArgumentParser(description='Create inflow file for LSM files and input directory.')

    # Define the command-line argument
    parser.add_argument('--lsmfile', type=str, help='LSM file containing RO variable')
    parser.add_argument('--inputsroot', type=str, help='Inputs directory')
    parser.add_argument('--inflowsroot', type=str, help='Inflows directory')
    args = parser.parse_args()

    # Access the parsed argument
    if not all([args.lsmfile, args.inputsroot, args.inflowsroot]):
        raise ValueError('Missing required arguments')

    # Create the inflow file for each input directory with the given LSM file
    create_inflow_file_for_all_inputs(args.lsmfile,
                                      args.inputsroot,
                                      args.inflowsroot,
                                      weight_table='weight_era5_721x1440.csv',
                                      comid_lat_lon_z='comid_lat_lon_z.csv')


if __name__ == '__main__':
    main()
