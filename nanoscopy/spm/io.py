import os
import glob
import numpy as np
import pandas as pd
import pySPM
import access2thematrix
import matplotlib.pyplot as plt
from pathlib import Path
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

        # Create SPMImage class object
        image = SPMImage(path)

        # Add records of important scan parameters
        direction = sxm.header['SCAN_DIR'][0][0]
        bias = float(sxm.header['Bias>Bias (V)'][0][0])
        setpoint_value = float(sxm.header['Z-CONTROLLER'][1][3])
        setpoint_unit = sxm.header['Z-CONTROLLER'][1][4]
        width = sxm.size['real']['x']
        height = sxm.size['real']['y']

        image.add_param('bias', bias)
        image.add_param('setpoint_value', setpoint_value)
        image.add_param('setpoint_unit', setpoint_unit) 
        image.add_param('width', width)
        image.add_param('height', height)

        # Get data and store in SPMImage class
        for channel in CHANNELS:
            for trace in TRACES:
                data = sxm.get_channel(channel, trace = trace).pixels
                data = data[~np.isnan(data)]
                image.add_data(channel, data)
                image.add_trace(channel, direction, trace)

        # Close the sxm file
        sxm.closefile() # Note: This method is from our modified version of SXM
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
        print(f'Processing {path}...', end="", flush=True)
        # Get I and Z mtrx file paths for this image.      
        mtrxs_ZI = {
            'Z': glob.glob(os.path.join(path, '*.Z_mtrx')),
            'Current': glob.glob(os.path.join(path, '*.I_mtrx'))
        }

        # Create SPMImage class object
        image = SPMImage(path)
        # Create mtrx class to access data
        mtrx = access2thematrix.MtrxData()

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

        # Get data and store in SPMImage class
        for channel, mtrx_path in mtrxs_ZI:
            rasters, message = mtrx.open(mtrx_path)
            for raster in rasters:
                data, message = mtrx.select_image(raster).data
                data = data[~np.isnan(data)]
                image.add_data(channel, data)
                
                direction, trace = raster.split('/')
                image.add_trace(channel, direction, trace)

        print('DONE')
        return image

    except Exception as error:
        raise

def read(path):
    """
    Imports tabular data from a text file.

    Inputs:
        path: A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path.
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        # If file, add to list as single element
        paths = [path]
    else:
        # if path points to a folder, get all paths in folder 
        paths = glob.glob(os.path.join(path, '*'), recursive=True)

    # Check if any files were found
    if len(paths) == 0:
        raise ValueError(FILES_NOT_FOUND_ERROR)

    images = []
    
    # Get all files with sxm extension
    sxms = list(filter(lambda p: p.endswith('sxm'), paths))
    sxm_images = list(map(read_sxm, sxms))
    images.extend(sxm_images)

    # Get all files with mtrx extension
    mtrxs = list(filter(lambda p: p.endswith('mtrx'), paths))
    # convert list of absolute paths to list of file names 
    mtrxs = list(map(lambda path: Path(path).stem, mtrxs))
    # make dictionary from list to obtain unique file names, 
    mtrxs = set(mtrxs)
    for file_name in mtrxs:
        # Combine Z and I mtrx files into single SPMimages
        partial_path = os.path.join(path, file_name)
        mtrx_images = read_mtrx(partial_path)
        images.extend(mtrx_images)

    return images

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