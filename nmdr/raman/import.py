import pandas as pd
import os
import glob

def read(path, src_format = None):
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