import pandas as pd

def load_data(path , translator = None):
    # General purpose loader.
    if translator == None:
        # Load data into a dataframe
        df = pd.read_csv(path, sep='\t', engine='python')
    
    elif translator in ['RenishawRaman','RenishawPL']:
        # Load data into a dataframe
        df = pd.read_csv(path, sep='\t', engine='python')
        
        if translator == 'RenishawRaman':
            # Make column labels more descriptive
            df.columns = ['Raman Shift','Intensity'] # Assign better names than defaults
        
            # Check to see if the Raman Shifts are listed in descending order. If the Raman shift is listed in descending order, order_check should be negative.
            order_check = df['Raman Shift'][1] - df['Raman Shift'][0]
        
        elif translator == 'RenishawPL':
            # Make column labels more descriptive
            df.columns = ['Photon Energy','Intensity'] # Assign better names than defaults
        
            # Check to see if the Raman Shifts are listed in descending order. If the Raman shift is listed in descending order, order_check should be negative.
            order_check = df['Photon Energy'][1] - df['Photon Energy'][0]

        # If the Raman shift is in descending order, flip the dataframe along the vertical direction to put it into ascending order.
        if order_check < 0:
            df = df.reindex(index=df.index[::-1]) # Reverse the order of the elements (put it in ascending order)
            df.reset_index(inplace=True, drop=True) # Reset the indexing of the dataframe to allow for normal slicing with the reversed order
    
    
    return df