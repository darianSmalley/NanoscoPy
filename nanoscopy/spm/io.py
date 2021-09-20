import os
import glob
import numpy as np
import pandas as pd
import pySPM
import access2thematrix
import matplotlib.pyplot as plt
from pathlib import Path
from . import SPMimage

spm_ext = tuple(['.sxm', '.Z_mtrx'])
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
        # Open SXM file
        sxm = pySPM.SXM(path)

        # Create SPMImage class object
        image = SPMimage(path)

        # Get data and store in SPMImage class
        for channel in ['Z' , 'Current']:
            im_forward = sxm.get_channel(channel , direction = 'forward').pixels
            im_backward = sxm.get_channel(channel , direction = 'backward').pixels

            image.add_data(im_forward , channel , 'Forward')
            image.add_data(im_backward , channel , 'Backward')

        # Add records of important scan parameters
        image.parameters['bias'] = float(sxm.header['Bias>Bias (V)'][0][0])
        image.parameters['setpoint_value'] = float(sxm.header['Z-CONTROLLER'][1][3])
        image.parameters['setpoint_unit'] = sxm.header['Z-CONTROLLER'][1][4]
        image.parameters['width'] = sxm.size['real']['x']
        image.parameters['height'] = sxm.size['real']['y']

        # Close the sxm file
        sxm.closefile() # Note: This method is from our modified version of SXM
        
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
        mtrx = access2thematrix.MtrxData()
        traces, message = mtrx.open(path)

         # Create SPMImage class object
        image = SPMimage(path)

        # Get data and store in SPMImage class
        for trace in traces:
            trace_image, message = mtrx.select_image(trace)
            trace_data = trace_image.data[~np.isnan(image.data)]
            image.add_data(trace_data , trace , 'Forward')

        # Retreive records of important scan parameters
        setpoint, setpoint_unit = mtrx.param['EEPA::Regulator.Setpoint_1']
        voltage, _ = mtrx.param['EEPA::GapVoltageControl.Voltage']
        scan_width, _ = mtrx.param['EEPA::XYScanner.Width']
        scan_height, _ = mtrx.param['EEPA::XYScanner.Height']

        # Add important scan parameters to SPMimage
        image.parameters['setpoint_value'] = setpoint
        image.parameters['setpoint_unit'] = setpoint_unit
        image.parameters['bias'] = voltage
        image.parameters['width'] = scan_width
        image.parameters['height'] = scan_height

        return image

    except Exception as error:
        print(message)
        raise

def read(path, filter = ''):
    """
    Imports tabular data from a text file.

    Inputs:
        path: A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path.
        filter: A string which specifies which files should be read.
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        # If file, add to list as single element
        paths = [path]          
    else:
        # if path points to a folder, 
        # get all paths in folder according to filter
        paths = glob.glob(os.path.join(path, f"*{filter}*"))

    # Check if any files were found
    if len(paths) == 0:
        raise ValueError(FILES_NOT_FOUND_ERROR)
    
    # read files with appropraite import function
    for path in paths: 
        if path.endswith(".sxm"):
            images = list(map(read_sxm, paths))
        elif path.endswith(".Z_mtrx"):
            images = list(map(read_mtrx, paths))
        else:
            raise ValueError(FILETYPE_ERROR)           

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