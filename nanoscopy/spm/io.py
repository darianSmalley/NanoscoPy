import os
import glob
import numpy as np
from datetime import datetime
import pandas as pd
import pySPM
import access2thematrix
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

from .SPMImage import SPMImage
from ..utilities import progbar

DEBUG = False

spm_ext = tuple(['.sxm', '.Z_mtrx'])
CHANNELS = ['Z' , 'Current']
TRACES = ['forward', 'backward']
FILES_NOT_FOUND_ERROR = "No files found."
FILETYPE_ERROR = "SUPPORTED FILETYPE NOT FOUND. Only sxm and Z_mtrx are supported."

def read_sxm(path):
    """
    Need to update this docstring

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
    
    Outputs:
        data: SPMImage. Contains the imported data. 
    """
    try:
        # Open SXM file
        sxm = pySPM.SXM(path)

        # Add records of important scan parameters
        filename = Path(path).with_suffix('').parts[-1]
        filename_parts = filename.split('_')
        probe = filename_parts[1]
        sample_id = '_'.join(filename_parts[0:2])
        rec_index = filename_parts[-1]

        direction = sxm.header['SCAN_DIR'][0][0]
        bias = float(sxm.header['BIAS'][0][0])
        setpoint = float(sxm.header['Z-CONTROLLER'][1][3])
        setpoint_unit = sxm.header['Z-CONTROLLER'][1][4]
        width = sxm.size['real']['x']
        height = sxm.size['real']['y']
        speed = sxm.header['Scan>speed forw. (m/s)'][0][0]
        time_per_line = sxm.header['SCAN_TIME'][0][0]

        # Get and correct channel list
        channels = sxm.header['Scan>channels'][0]
        channels = '_'.join(channels)
        channels = channels.split(';')

        # Get and convert date/time stamp
        date = sxm.header['REC_DATE'][0][0]
        time = sxm.header['REC_TIME'][0][0]
        datetime_string = date + ' ' + time
        datetime_object = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S')

        dfs = []
        for channel in CHANNELS:
            for trace in TRACES:
                sxm_data = sxm.get_channel(channel, direction = trace).pixels
                nan_mask = ~np.isnan(sxm_data)
                img_data = np.where(nan_mask, sxm_data, 0)
                # pySPM seems to flip the image to put the origin in the bottom left corner
                # flip again to correct
                if direction == 'up':
                    img_data = np.flipud(img_data)
                
                # if (channel == 'Z') and (trace == 'forward'):
                #     plt.imshow(img_data)

                data = [
                    sample_id,
                    rec_index, 
                    probe,
                    channel,
                    direction,
                    trace,
                    setpoint,
                    bias,
                    width,
                    height,
                    time_per_line,
                    datetime_object.isoformat(),
                    path,
                    img_data
                ]
                
                formatted_data = dict(zip(SPMImage.data_headers, data))
                df = pd.DataFrame([formatted_data])
                dfs.append(df)

        # Append list of dataframes into single dataframe
        dataframe = pd.concat(dfs, ignore_index=True)
        # Create SPMImage to store data and metadata
        image = SPMImage(dataframe, sxm.header)

        return image

    except Exception as error:
        print('Error detected:', error)
        raise

def read_sxm_v1(path):
    """
    Need to update this docstring

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
    
    Outputs:
        data: SPMImage. Contains the imported data. 
    """
    try:
        # Open SXM file
        sxm = pySPM.SXM(path)

        # Add records of important scan parameters
        filename = Path(path).with_suffix('').parts[-1]
        filename_parts = filename.split('_')
        probe = filename_parts[1]
        sample_id = filename_parts[2]
        rec_index = filename_parts[-1]

        scan_direction = sxm.header['SCAN_DIR'][0][0]
        bias = float(sxm.header['BIAS'][0][0])
        setpoint_value = float(sxm.header['Z-CONTROLLER'][1][3])
        setpoint_unit = sxm.header['Z-CONTROLLER'][1][4]
        width = sxm.size['real']['x']
        height = sxm.size['real']['y']

        # Get and correct channel list
        channels = sxm.header['Scan>channels'][0]
        channels = '_'.join(channels)
        channels = channels.split(';')

        # Get and convert date/time stamp
        date = sxm.header['REC_DATE'][0][0]
        time = sxm.header['REC_TIME'][0][0]
        datetime_string = date + ' ' + time
        datetime_object = datetime.strptime(datetime_string, '%d.%m.%Y %H:%M:%S')

        # Create SPMImage class object
        image = SPMImage(path)
        image.add_param('bias', bias)
        image.add_param('channels', channels)
        image.add_param('date_time' , datetime_object)
        image.add_param('width', width)
        image.add_param('height', height)
        image.add_param('setpoint_value', setpoint_value)
        image.add_param('setpoint_unit', setpoint_unit) 
        image.add_param('scan_direction', scan_direction)
        image.add_param('probe', probe)
        image.add_param('sample_id', sample_id)
        image.add_param('rec_index', rec_index)

        image.set_headers(sxm.header)

        # Get data and store in SPMImage class
        for channel in CHANNELS:
            for trace in TRACES:
                data = sxm.get_channel(channel, direction = trace).pixels
                nan_mask = ~np.isnan(data)
                data = np.where(nan_mask, data, 0)
                image.add_data(channel, data)
                image.add_trace(channel, scan_direction, trace)

        return image

    except Exception as error:
        print('Error detected:', error)
        raise

def read_mtrx_v1(path):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: numpy array. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    try: 
        # TODO: check if path has extension and remove it

        # Get I and Z mtrx file paths for this image.   
        Z_path = f'{path}.Z_mtrx'
        I_path = f'{path}.I_mtrx'
        mtrx_channels = {
            'Z': glob.glob(Z_path),
            'Current': glob.glob(I_path)
        }
        
        print('Found the following mtrx files in path:')
        # pprint.pprint(mtrx_channels)
        print(f'Processing {Path(path).stem}...', end="", flush=True)

        data = {
            'channels': [],
            'paths':   [],
            'setpoints': [],
            'voltages': [],
            'widths': [],
            'heights': [],
            'datetimes': [],
            'directions': [],
            'traces': [],
            'images': []
        }

        # Create SPMImage class object
        image = SPMImage(path)
        # Get data and store in SPMImage class
        for channel, channel_paths in mtrx_channels.items():
            for channel_path in channel_paths:
                # Create mtrx class to access data
                mtrx = access2thematrix.MtrxData()    
                rasters, message = mtrx.open(channel_path)

                # Retreive records of important scan parameters
                setpoint, setpoint_unit = mtrx.param['EEPA::Regulator.Setpoint_1']
                voltage, _ = mtrx.param['EEPA::GapVoltageControl.Voltage']
                scan_width, _ = mtrx.param['EEPA::XYScanner.Width']
                scan_height, _ = mtrx.param['EEPA::XYScanner.Height']

                # Get and convert date/time stamp
                datetime_string = mtrx.param['BKLT']
                datetime_object = datetime.strptime(datetime_string, '%A, %d %B %Y %H:%M:%S')

                # Add important scan parameters to SPMimage
                image.add_param('setpoint_value', setpoint)
                image.add_param('setpoint_unit', setpoint_unit)
                image.add_param('bias', voltage)
                image.add_param('width', scan_width)
                image.add_param('height', scan_height)
                image.add_param('date_time' , datetime_object)
                image.set_headers(mtrx.param)

                for _, raster in rasters.items():
                    trace, direction = raster.split('/')
                    image.add_trace(channel, direction, trace)

                    mtrx_image, message = mtrx.select_image(raster)
                    nan_mask = ~np.isnan(mtrx_image.data)
                    mtrx_image = np.where(nan_mask, mtrx_image.data, 0)
                    image.add_data(channel, np.array(mtrx_image.data))

                    data['channels'].append(channel)
                    data['directions'].append(direction)
                    data['traces'].append(trace)
                    data['setpoints'].append(setpoint)
                    data['voltages'].append(voltage)
                    data['widths'].append(scan_width)
                    data['heights'].append(scan_height)
                    data['datetimes'].append(datetime_string)
                    data['paths'].append(channel_path)
                    data['images'].append(np.array(mtrx_image.data))
        
        dataframe = pd.DataFrame(data)
        image.add_dataframe(dataframe)
        return image

    except Exception as error:
        raise

def read_mtrx_v2(path):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: numpy array. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    try: 
        if DEBUG: print(path)
        # Get I and Z mtrx file paths for this image.   
        Z_path = f'{path}.Z_mtrx'
        I_path = f'{path}.I_mtrx'
        mtrx_channels = {
            'Z': glob.glob(Z_path),
            'Current': glob.glob(I_path)
        }
        if DEBUG: pprint.pprint(mtrx_channels)
        probe = 'WTip'

        # Add records of important scan parameters
        full_path = Path(path).with_suffix('')
        material = full_path.parts[-4]
        growth_params = full_path.parts[-3]
        scan_date = full_path.parts[-2]
        filename = full_path.parts[-1]
        filename_parts = filename.split('--')
        rec_index = filename_parts[-1]
        sample_id = '_'.join([material, growth_params])

        dfs = []
        # Get data and store in SPMImage class
        for channel, channel_paths in mtrx_channels.items():
            for channel_path in channel_paths:
                # Create mtrx class to access data
                mtrx = access2thematrix.MtrxData()    
                rasters, message = mtrx.open(channel_path)

                # Retreive records of important scan parameters
                setpoint, setpoint_unit = mtrx.param['EEPA::Regulator.Setpoint_1']
                bias, _ = mtrx.param['EEPA::GapVoltageControl.Voltage']
                width, _ = mtrx.param['EEPA::XYScanner.Width']
                height, _ = mtrx.param['EEPA::XYScanner.Height']
                relocation_step, _ = mtrx.param['EEPA::XYScanner.Relocation_Step_Limit']
                relocation_time, _ = mtrx.param['EEPA::XYScanner.Relocation_Time_Limit']
                raster_time, _ = mtrx.param['EEPA::XYScanner.Raster_Time']
                speed = width/raster_time

                # Get and convert date/time stamp
                # datetime_string = mtrx.param['BKLT']
                # datetime_object = datetime.strptime(datetime_string, '%A, %d %B %Y %H:%M:%S')
                datetime_object = datetime.strptime(scan_date, '%Y-%m-%d')

                # Add important scan parameters to SPMimage
                for _, raster in rasters.items():
                    trace, direction = raster.split('/')
                    mtrx_image, message = mtrx.select_image(raster)
                    nan_mask = ~np.isnan(mtrx_image.data)
                    mtrx_image = np.where(nan_mask, mtrx_image.data, 0)

                    data = [
                        sample_id,
                        rec_index,
                        probe,
                        channel,
                        direction,
                        trace,
                        setpoint,
                        bias,
                        width,
                        height,
                        raster_time,
                        datetime_object,
                        path,
                        mtrx_image
                    ]

                    formatted_data = dict(zip(SPMImage.data_headers, data))
                    df = pd.DataFrame([formatted_data])
                    dfs.append(df)

        # Append list of dataframes into single dataframe
        dataframe = pd.concat(dfs, ignore_index=True)
        # Create SPMImage to store data and metadata
        image = SPMImage(dataframe, mtrx.param)

        return image

    except Exception as error:
        raise

def read_mtrx(path):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: numpy array. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    try: 
        if DEBUG: print(path)

        # Get I and Z mtrx file paths for this image.   
        Z_path = Path(path).with_suffix('.Z_mtrx')
        I_path = Path(path).with_suffix('.I_mtrx')
        mtrx_channels = {
            'Z': glob.glob(str(Z_path)),
            'Current': glob.glob(str(I_path))
        }

        # Add records of important scan parameters
        full_path = Path(path).with_suffix('')
        material = full_path.parts[-4]
        growth_params = full_path.parts[-3]
        scan_date = full_path.parts[-2]
        sample_id = '_'.join([material, growth_params])

        filename = full_path.parts[-1]
        filename_parts = filename.split('--')
        rec_index = filename_parts[-1]

        probe = 'WTip'

        dfs = []
        # Get data and store in SPMImage class
        for channel, channel_paths in mtrx_channels.items():
            for channel_path in channel_paths:
                # Create mtrx class to access data
                mtrx = access2thematrix.MtrxData()    
                rasters, message = mtrx.open(channel_path)

                # Retreive records of important scan parameters
                setpoint, setpoint_unit = mtrx.param['EEPA::Regulator.Setpoint_1']
                bias, _ = mtrx.param['EEPA::GapVoltageControl.Voltage']
                width, _ = mtrx.param['EEPA::XYScanner.Width']
                height, _ = mtrx.param['EEPA::XYScanner.Height']
                relocation_step, _ = mtrx.param['EEPA::XYScanner.Relocation_Step_Limit']
                relocation_time, _ = mtrx.param['EEPA::XYScanner.Relocation_Time_Limit']
                raster_time, _ = mtrx.param['EEPA::XYScanner.Raster_Time']
                speed = width/raster_time

                # Get and convert date/time stamp
                # datetime_string = mtrx.param['BKLT']
                # datetime_object = datetime.strptime(datetime_string, '%A, %d %B %Y %H:%M:%S')
                datetime_object = datetime.strptime(scan_date, '%Y-%m-%d')

                # Add important scan parameters to SPMimage
                for _, raster in rasters.items():
                    trace, direction = raster.split('/')
                    mtrx_image, message = mtrx.select_image(raster)
                    nan_mask = ~np.isnan(mtrx_image.data)
                    mtrx_image = np.where(nan_mask, mtrx_image.data, 0)

                    data = [
                        sample_id,
                        rec_index,
                        probe,
                        channel,
                        direction,
                        trace,
                        setpoint,
                        bias,
                        width,
                        height,
                        raster_time,
                        datetime_object.isoformat(),
                        path,
                        mtrx_image
                    ]

                    formatted_data = dict(zip(SPMImage.data_headers, data))
                    df = pd.DataFrame([formatted_data])
                    dfs.append(df)

        # Append list of dataframes into single dataframe
        dataframe = pd.concat(dfs, ignore_index=True)
        # Create SPMImage to store data and metadata
        image = SPMImage(dataframe, mtrx.param)

        return image

    except Exception as error:
        raise

def read(path, filename_filter = ''):
    """
    Imports microscopy image data from Nanonis (.sxm) and Omicron (.Z_mtrx, .I_mtrx) files.

    Inputs:
        path: A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path.
    
    Outputs:
        data: list of SPMiamges. Contains the imported data and select metadata. 
    """
    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        # If file, check if sxm or mtrx
        if path.endswith('sxm'):
            data = [read_sxm(path)]
        elif path.endswith('_mtrx'):
            # clean path of extensions
            path = Path(path).with_suffix('')
            # add to list as single image
            data = [read_mtrx(path)]
        else:
            raise ValueError(FILETYPE_ERROR)
    else:
        # if path points to a folder, get all paths in folder according to filter
        paths = glob.glob(os.path.join(path, f'**/*{filename_filter}*.*'), recursive=True)
        paths = sorted(paths)
        n = len(paths)
        
        # Check if any files were found
        if n == 0: raise ValueError(FILES_NOT_FOUND_ERROR)

        if DEBUG: 
            print(n, 'paths found.')
            pprint.pprint(paths)

        # Create empty data list
        data = []
        
        # Get all files with sxm extension and extend data list
        sxms = list(filter(lambda p: p.endswith('sxm'), paths))
        if len(sxms):
            print(f'Reading {len(sxms)} sxm files...', end="", flush=True)
            sxm_data = list(map(read_sxm, sxms))
            data.extend(sxm_data)
            print('DONE')

        # Get all files with Z or I mtrx extension and extend data list
        mtrxs = list(filter(lambda p: p.endswith('Z_mtrx'), paths))
        if len(mtrxs):
            # convert list of absolute paths to set of unique file names 
            # mtrxs = set(map(lambda path: Path(path).stem, mtrxs))
            print(f'Reading {len(mtrxs)} mtrx files...', end="", flush=True)
            
            # Combine Z and I mtrx files into single SPMimages
            for file_name in mtrxs:
                # file_path = os.path.join(path, file_name)
                mtrx_image = read_mtrx(file_name)
                data.append(mtrx_image)

            print('DONE')

    return data

def export_images(images, dst_paths = ['./'], ext = 'jpg', 
               scan_dir = 'up', cmap = 'gray'):
    """
    Exports image from numpy array

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if not isinstance(images, list): images = [images]
    if not isinstance(dst_paths, list): dst_paths = [dst_paths]

    for image, dst_path in zip(images, dst_paths):
        origin = 'lower' if scan_dir == 'up' else 'upper'
        plt.imsave(dst_path, image, cmap=cmap, origin=origin)

def export_metadata(images, dst_path = './'):
    metadata = [image.dataframe.drop('image', axis=1) for image in images]
    metadata = pd.concat(metadata)
    metadata.to_csv(os.path.join(dst_path, 'metadata_out.csv'), index=False)