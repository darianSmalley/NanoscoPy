import os
from shutil import copy2
from tkinter import Tk, filedialog, Label

def pad_filename_with_zeros(filepath, file_index_delim = '_' , pad_digits = 4):
    """
    filepath: string. Contains path to file whose name is to be padded
    file_index_delim: string. Contains character which immediately precedes file index number.
    pad_digits: int. Specifies the number of digits to be present in the final file number after padding.
    """
    # Find the last period, which should denote the beginning of the file extension
    final_period = filepath.rfind('.')
    # Determine the file extension
    extension = filepath[final_period:]
    # Find the final ocurrence of the file index delimiter
    final_underscore = filepath.rfind(file_index_delim)
    # Determine the file index
    file_number = filepath[final_underscore+1:final_period]
    # Generate the new file index, padded to specified number of digits
    new_file_number = '0'*(pad_digits-len(file_number))+file_number
    # Generate new file name, including padded file index.
    new_filename = filepath[:final_underscore] + '_' + new_file_number + extension
    return new_filename

def rename_with_padding(source_folder_path , destination_folder_path , copy_files = True , file_index_delim = '_' , pad_digits = 4):
    """
    source_folder_path: string. Contains the directory holding the files with names to be padded.
    destination_folder_path: string. Contains the directory into which the renamed files are to be saved.
    copy_files: bool. A value of true signifies that the files to be renamed should copied to the destination directory. False signifies that the files will be renamed in place.
    file_index_delim: string. Contains character which immediately precedes file index number.
    pad_digits: int. Specifies the number of digits to be present in the final file number after padding.
    """
    # Iterate through the entire directory.
    for file in os.listdir(source_folder_path):
        # Generate new filename with index padding.
        new_filename = pad_filename_with_zeros(file , file_index_delim = file_index_delim , pad_digits = pad_digits)
        if copy_files == True:
            # Copy and rename files with padding if copy_files is set to true.
            copy2( os.path.join(source_folder_path, file) , os.path.join(destination_folder_path , new_filename ))
        else:
            # Rename files in place with padding if copy_files is not set to True
            os.rename( os.path.join(source_folder_path, file) , os.path.join(destination_folder_path , new_filename ))

def sort_data_ascending(data, indep_variable_name):
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

def dialog_askdirectory():
    """
    Prompts the user to select a directory and returns the path for that directory.

    Output:
        folder_path: string. The full path to the selected directory.
    """
    root = Tk()
    Label(root, text="Select data folder")
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def dialog_askfilename():
    """
    Prompts the user to select a file and returns the path for that file.

    Output:
        file_path: string. The full path to the selected file.
    """
    root = Tk()
    Label(root, text="Select data folder")
    file_path = filedialog.askopenfilename() 
    root.destroy()
    return  file_path
