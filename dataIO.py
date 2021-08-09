import pandas as pd
import os

def determine_metadata_lines(path , source = 'Nanonis'):
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

def sort_data_ascending(data , indep_variable_name):
    """
    Checks whether the input data is sorted in ascending order.
    
    Inputs:
        data: DataFrame. Contains the data to be sorted.
        indep_variable_name: string. Specifies the column name in data which is the independent variable to be used for sorting.
    
    Outputs:
        data: DataFrame. Contains the data, sorted in ascending order.
    """
    if data[indep_variable_name][1] < data[indep_variable_name][0]:
        data = data.reindex(index = data.index[::-1]) # Reverse the order of the elements (put it in ascending order)
        data.reset_index(inplace = True, drop = True) # Reset the indexing of the dataframe to allow for normal slicing with the reversed order
    return data

def load_data(path , translator = None):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        translator: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)
    
    Outputs:
        data: DataFrame. Contains the imported data. Most of the translators also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if translator == None:
        # Load data into a dataframe
        data = pd.read_csv(path, sep='\t', engine='python')
    
    elif translator in ['RenishawRaman','RenishawPL']:
        # Load data into a dataframe
        data = pd.read_csv(path, sep='\t', engine='python')
        
        if translator == 'RenishawRaman':
            # Make column labels more descriptive
            data.columns = ['Raman Shift','Intensity'] # Assign better names than defaults

            # Sort the data so that Raman Shift values are in ascending order.
            data = sort_data_ascending(data , 'Raman Shift')
        
        elif translator == 'RenishawPL':
            # Make column labels more descriptive
            data.columns = ['Photon Energy','Intensity'] # Assign better names than defaults
        
            # Sort the data so that Photon Energy values are in ascending order.
            data = sort_data_ascending(data , 'Photon Energy')
    
    elif translator in ['Nanonis-IV' , 'Nanonis-IZ']:
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
        if translator == 'Nanonis-IV':
            data = sort_data_ascending(data , 'Bias calc (V)')
        elif translator == 'Nanonis-IZ':
            data = sort_data_ascending(data , 'Z rel (m)')
    
    return data