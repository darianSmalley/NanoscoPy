import os
import glob
import numpy as np
import pandas as pd
import pySPM
import access2thematrix

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
    # # Determine if the path was specified as a list of path snipets
    # if isinstance(path, list):
    #     # Use list unpacking to separate elements into multiple arguments, then join them into a proper path.
    #     filepath = os.path.join(*path)
    
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
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: numpy array. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    data = pySPM.SXM(path)

    try:
        # Get Z scan pixels
        image = data.get_channel('Z').pixels
        # Convert to numpy array
        image = np.asmatrix(image)
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
    mtrx = access2thematrix.MtrxData()

    try: 
        traces, message = mtrx.open(path)
        image, message = mtrx.select_image(traces[0])
        image = image.data[~np.isnan(image.data)]
        return image
    
    except Exception as error:
        print(message)
        raise

def read_dir(directory, signal = None, filter = '', ext = 'dat'):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    paths = glob.glob(os.path.join(directory, f"*{filter}*.{ext}"))
    
    if len(paths) == 0:
        raise ValueError("No files found.")

    if paths[0].endswith(".dat"):
        data = list(map(lambda p: read_spectrum(p, signal), paths))
    
    elif paths[0].endswith(".sxm"):
        data = list(map(read_sxm, paths))
            
    elif paths[0].endswith(".Z_mtrx"):
        data = list(map(read_mtrx, paths))
        
    elif paths[0].endswith(".3ds"):
        pass
    else:
        raise ValueError("SUPPORTED FILETYPE NOT FOUND. Only dat, sxm, Z_mtrx, and 3ds are supported.")

    return data