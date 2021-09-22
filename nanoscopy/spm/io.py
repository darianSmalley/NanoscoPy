import os
import glob
import numpy as np
import pandas as pd
import pySPM
import access2thematrix
import matplotlib.pyplot as plt
from pathlib import Path
import pprint
from .SPMImage import SPMImage

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
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: numpy array. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    try:
        print(f'Processing {path}...', end="", flush=True)
        # Open SXM file
        sxm = pySPM.SXM(path)

        # Add records of important scan parameters
        channels = sxm.header['Scan>channels'][0]
        scan_direction = sxm.header['SCAN_DIR'][0][0]
        bias = float(sxm.header['BIAS'][0][0])
        setpoint_value = float(sxm.header['Z-CONTROLLER'][1][3])
        setpoint_unit = sxm.header['Z-CONTROLLER'][1][4]
        width = sxm.size['real']['x']
        height = sxm.size['real']['y']

        # Create SPMImage class object
        image = SPMImage(path)
        image.add_param('bias', bias)
        image.add_param('setpoint_value', setpoint_value)
        image.add_param('setpoint_unit', setpoint_unit) 
        image.add_param('width', width)
        image.add_param('height', height)
        image.add_param('channels', channels)
        image.add_param('scan_direction', scan_direction)
        image.set_headers(sxm.header)

        # Get data and store in SPMImage class
        for channel in CHANNELS:
            for trace in TRACES:
                data = sxm.get_channel(channel, direction = trace).pixels
                data = data[~np.isnan(data)]
                image.add_data(channel, data)
                image.add_trace(channel, scan_direction, trace)

        print('DONE')
        return image

    except Exception as error:
        print('Error detected:', error)
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
        # TODO: check if path has extension and remove it

        # Get I and Z mtrx file paths for this image.   
        Z_path = f'{path}.Z_mtrx'
        I_path = f'{path}.I_mtrx'
        mtrx_channels = {
            'Z': glob.glob(Z_path),
            'Current': glob.glob(I_path)
        }
        
        print('Found the following mtrx files in path:')
        pprint.pprint(mtrx_channels)
        print(f'Processing {Path(path).stem}...', end="", flush=True)

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

                # Add important scan parameters to SPMimage
                image.add_param('setpoint_value', setpoint)
                image.add_param('setpoint_unit', setpoint_unit)
                image.add_param('bias', voltage)
                image.add_param('width', scan_width)
                image.add_param('height', scan_height)
                image.set_headers(mtrx.param)
                
                for index, raster in rasters.items():
                    mtrx_image, message = mtrx.select_image(raster)
                    # data = mtrx_image.data[~np.isnan(mtrx_image.data)]
                    image.add_data(channel, mtrx_image.data)
                    
                    trace, direction = raster.split('/')
                    image.add_trace(channel, direction, trace)
            
        print('DONE')
        return image

    except Exception as error:
        raise

def read(path):
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
        # if path points to a folder, get all paths in folder 
        paths = glob.glob(os.path.join(path, '*'), recursive=True)

        # Check if any files were found
        if len(paths) == 0:
            raise ValueError(FILES_NOT_FOUND_ERROR)

        # Create empty data list
        data = []
        
        # Get all files with sxm extension and extend data list
        sxms = list(filter(lambda p: p.endswith('sxm'), paths))
        if len(sxms):
            sxm_data = list(map(read_sxm, sxms))
            data.extend(sxm_data)

        # Get all files with mtrx extension and extend data list
        mtrxs = list(filter(lambda p: p.endswith('_mtrx'), paths))
        if len(mtrxs):
            # convert list of absolute paths to set of unique file names 
            mtrxs = set(map(lambda path: Path(path).stem, mtrxs))

            # Combine Z and I mtrx files into single SPMimages
            for file_name in mtrxs:
                file_path = os.path.join(path, file_name)
                mtrx_image = read_mtrx(file_path)
                data.append(mtrx_image)

    return data

def export_spm(images, dst_paths = ['./'], ext = 'jpeg', 
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

    for image, path in zip(images, dst_paths):
        path_without_ext = Path(path).with_suffix('')
        dst = f'{path_without_ext}.{ext}'
        origin = 'lower' if scan_dir == 'up' else 'upper'
        plt.imsave(dst, image, cmap=cmap, origin=origin)