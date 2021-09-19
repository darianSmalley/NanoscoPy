import os
import glob
import numpy as np
import pandas as pd
import pySPM
import access2thematrix
import matplotlib.pyplot as plt
from pathlib import Path
import SPMimage

FILES_NOT_FOUND_ERROR = "No files found."
FILETYPE_ERROR = "SUPPORTED FILETYPE NOT FOUND. Only dat, sxm, Z_mtrx, and 3ds are supported."

def determine_metadata_lines(path, source = 'Nanonis'):
    """
    Searches the beginning lines of a file containing tabular data to determine the number of lines at the beginning of the file contain metadata.
    
    Inputs:
        path: string. Specifies the full file path of the file to be examined. Should include the file extension.
        source: string. Specifies a family of instrument which generated the file to be examined. This determines how the algorithm locates the end of the metadata.
    
    Outputs:
        metadata_end: int. Specifies the row index corresponding to the end of the metadata at the beginnong of the data file.
    """
    if source == 'Nanonis':
        # Determine if the path was specified as a list of path snipets
        if isinstance(path, list): 
            # Use list unpacking to separate elements into multiple arguments, then join them into a proper path.
            filepath = os.path.join(*path)
        
        # Load only the first column of data.
        data_column = pd.read_csv(path , sep = '\t' , usecols = [0] , header = None)
        
        # Find the first match for '[DATA]' in the file. This should always be one row before the beginning of the data.
        metadata_end = data_column[ data_column[0] == "[DATA]"].index[0]
    return metadata_end

def read_spectrum(path, signal = None):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        signal: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the signals also ensure that the data is sorted such that the independent variable is is ascending order.
    """  
    # Determine the number of header rows in the file.
    metadata_end = determine_metadata_lines(path , source = 'Nanonis')
    
    # Load only the data portion (with column names), skipping the header.
    data = pd.read_csv(path , sep = '\t' , header = 1 , skiprows = metadata_end)
    
    # Drop any rows that contain NaN as an element.
    data = data.dropna()
    
    # Sort the data in ascending order.
    if signal: 
        data = data.sort_values(by=[signal])

    return data

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
        data = pySPM.SXM(path)

        # Create SPMImage class object
        image = SPMimage()

        # Get data and store in SPMImage class
        for channel in ['Z' , 'Current']:
            im_forward = data.get_channel(channel , direction = 'forward').pixels
            im_backward = data.get_channel(channel , direction = 'backward').pixels

            image.add_data(im_forward , channel , 'Forward')
            image.add_data(im_backward , channel , 'Backward')

        # Add records of important scan parameters
        image.parameters['bias'] = float(data.header['Bias>Bias (V)'][0][0])
        image.parameters['setpoint_value'] = float(data.header['Z-CONTROLLER'][1][3])
        image.parameters['setpoint_unit'] = data.header['Z-CONTROLLER'][1][4]
        image.parameters['width'] = data.size['real']['x']
        image.parameters['height'] = data.size['real']['y']

        # Close the sxm file
        data.closefile() # Note: This method is from our modified version of SXM
        
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
        image, message = mtrx.select_image(traces[0])
        image = image.data[~np.isnan(image.data)]
        return image
    
    except Exception as error:
        print(message)
        raise

def read_spectra(paths, signal = None):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if paths[0].endswith(".dat"):
        data = list(map(lambda p: read_spectrum(p, signal), paths))

    elif paths[0].endswith(".3ds"):
        pass

    return data

def read_spm(paths):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if paths[0].endswith(".sxm"):
        data = list(map(read_sxm, paths))
            
    elif paths[0].endswith(".Z_mtrx"):
        data = list(map(read_mtrx, paths))

    return data

def read(path, filter = '', signal = None):
    """
    Imports tabular data from a text file.

    Inputs:
        path: A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path.
        filter: A string which specifies which files should be read.
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    spm_ext = tuple(['.sxm', '.Z_mtrx'])
    sts_ext = tuple(['.dat', '.3ds'])

    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        # If file ends with supported spm file extension,
        # read file with appropraite import function
        if path.endswith('.sxm'):
            data = read_sxm(path)
        elif path.endswith('.Z_mtrx'):
            data = read_mtrx(path)
        elif path.endswith('.dat'):
            data = read_spectrum(path, signal)
        elif path.endswith('.3ds'):
            pass
        else:
            raise ValueError(FILETYPE_ERROR)

        return data, path            
    # if path points to a folder, 
    # get all paths in folder according to filter
    else:
        paths = glob.glob(os.path.join(path, f"*{filter}*"))

        # Check if any files were found
        if len(paths) == 0:
            raise ValueError(FILES_NOT_FOUND_ERROR)
            
        # Read files with appropriate import function
        elif paths[0].endswith(spm_ext):
            data = read_spm(paths)
        elif paths[0].endswith(sts_ext):
            data = read_spectra(paths, signal)
        else:
            raise ValueError(FILETYPE_ERROR)

        return data, paths
    
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