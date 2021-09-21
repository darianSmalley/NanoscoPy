import pandas as pd
import os
import glob

sts_ext = tuple(['.dat', '.3ds'])
FILES_NOT_FOUND_ERROR = "No files found."
FILETYPE_ERROR = "SUPPORTED FILETYPE NOT FOUND. Only dat and 3ds are supported."

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

def read_raman(path, src_format = None):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if src_format == None:
        # Load data into a dataframe
        data = pd.read_csv(path, sep='\t', engine='python')
    
    elif src_format in ['RenishawRaman','RenishawPL']:
        # Load data into a dataframe
        data = pd.read_csv(path, sep='\t', engine='python')
        
        if src_format == 'RenishawRaman':
            # Make column labels more descriptive
            data.columns = ['Raman Shift','Intensity'] # Assign better names than defaults

            # Sort the data so that Raman Shift values are in ascending order.
            data = data.sort_values(by=['Raman Shift'])
        
        elif src_format == 'RenishawPL':
            # Make column labels more descriptive
            data.columns = ['Photon Energy','Intensity'] # Assign better names than defaults
        
            # Sort the data so that Photon Energy values are in ascending order.
            data = data.sort_values(by=['Photon Energy'])
    
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
    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        # If file ends with supported spm file extension,
        # read file with appropraite import function
        if path.endswith('.sxm'):
            data = read_sxm(path)
        elif path.endswith('.Z_mtrx'):
            data, metadata = read_mtrx(path)
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