import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd

from .spectrum import Spectrum, STS
from ..utilities import progbar

sts_ext = tuple([".dat", ".3ds"])
FILES_NOT_FOUND_ERROR = "No files found."
FILETYPE_ERROR = "SUPPORTED FILETYPE NOT FOUND. Only dat and 3ds are supported."


def determine_metadata_lines(path: str) -> int:
    """
    Searches the beginning lines of a file containing tabular data to determine the number of lines at the beginning of the file contain metadata.

    Args:
        path (str): Specifies the full file path of the file to be examined. Should include the file extension.

    Returns:
        metadata_end (int): Specifies the row index corresponding to the end of the metadata at the beginnong of the data file.
    """

    # Load only the first column of data.
    data_column = pd.read_csv(path, sep="\t", usecols=[0], header=None)

    # Find the first match for '[DATA]' in the file. This should always be one row before the beginning of the data.
    metadata_end = data_column[data_column[0] == "[DATA]"].index[0]

    return metadata_end


def get_metadata(path, metadata_end) -> pd.DataFrame:
    """
    Searches the beginning lines of a file containing tabular data to determine the number of lines at the beginning of the file contain metadata.

    Args:
        path (str): Specifies the full file path of the file to be examined. Should include the file extension.

    Returns:
        metadata (pd.DataFrame): panadas DataFrame containing all metadata found in the STS file header.
    """
    metadata = pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1],
        names=["Property", "Value"],
        skiprows=1,
        nrows=metadata_end - 1,
    )
    metadata = metadata.set_index("Property")
    return metadata


def read_STS(path: str, source: str = "Nanonis", sep: str = "\t") -> STS:
    """
    Reads tabular STS data from a text file.

    Args:
        path (str): Path string of the file to be read.
        sep (str): String dilimiter used during reading

    Returns:
        spectrum (STS): Class contains the imported data.
    """
    # Load data into a dataframe

    if source == "Nanonis":
        # Determine the number of header rows in the file.
        metadata_end = determine_metadata_lines(path)

        # Get metadata as a pandas DataFrame
        metadata = get_metadata(path, metadata_end)

        # Load only the data portion (with column names), skipping the header.
        data = pd.read_csv(path, sep=sep, header=1, skiprows=metadata_end)

    else:
        metadata = None
        data = pd.read_csv(path, sep=sep, engine="python")

    # Drop any columns that contain all NaN values
    data = data.dropna(axis=1, how="all")

    # Drop any rows that contain any NaN values
    data = data.dropna(axis=0, how="any")

    spectrum = STS(data, metadata, filepath=path)

    return spectrum


def read_raman(path, source="RenishawRaman"):
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        src_format: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)

    Outputs:
        data: DataFrame. Contains the imported data. Most of the src_formats also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    # Load data into a dataframe
    data = pd.read_csv(path, sep="\t", engine="python")

    if source == "RenishawRaman":
        # Make column labels more descriptive
        # Assign better names than defaults
        data.columns = ["Raman Shift", "Intensity"]

        # Sort the data so that Raman Shift values are in ascending order.
        data = data.sort_values(by=["Raman Shift"])

    elif source == "RenishawPL":
        # Make column labels more descriptive
        # Assign better names than defaults
        data.columns = ["Photon Energy", "Intensity"]

        # Sort the data so that Photon Energy values are in ascending order.
        data = data.sort_values(by=["Photon Energy"])

    spectrum = Spectrum(data)

    return spectrum


def read_numpy(path):
    data = np.load(path)
    data[[0, 1]] = data[[1, 0]]
    data_df = pd.DataFrame(data.T)
    spectrum = STS(data_df)
    return spectrum


def read_spectrum(path, source="Nanonis", sep="\t") -> STS:
    """
    Imports tabular data from a text file.

    Inputs:
        path: string. Specifies the full file path (including file extension) of the file to be imported.
        signal: string. Specifies which (if any) specialized importing routines should be used to prepare the data (e.g. to skip metadata at the beginning of a file)

    Outputs:
        data: DataFrame. Contains the imported data. Most of the signals also ensure that the data is sorted such that the independent variable is is ascending order.
    """
    if source == "Nanonis":
        spectrum = read_STS(path, source)

    elif source in ["RenishawRaman", "RenishawPL"]:
        spectrum = read_raman(path, source)
    else:
        # data = pd.read_csv(path, sep=sep, engine='python')
        # spectrum = Spectrum(data, filepath=path)
        spectrum = read_STS(path, source, sep)
    return spectrum


def read_spectra(paths: List[str], source: str) -> List[STS]:
    """
    Imports tabular data from a text file.

    Args:
        path: string. Specifies the full file path (including file extension) of the file to be imported.

    Returns:
        data (List[STS]): Contains the imported data as STS objects.
    """

    spectra = []
    for i, path in enumerate(paths):
        progbar(
            i + 1,
            len(paths),
            10,
            "Reading spectra...Done" if i + 1 == len(paths) else "Reading spectra...",
        )

        if path.endswith(".dat"):
            spectrum = read_spectrum(path, source)
        elif path.endswith(".csv"):
            spectrum = read_spectrum(path, source=None, sep=",")
        elif path.endswith(".npy"):
            spectrum = read_numpy(path)
        else:
            raise ValueError(FILETYPE_ERROR)

        spectra.append(spectrum)

    return spectra


def read(path: str, filter: str = "", source: str = "Nanonis") -> List[STS]:
    """
    Imports tabular data from a text file.

    Args:
        path (str): A path-like object representing a file system path. A path-like object is either a string or bytes object representing a path.
        filter (str): A string which specifies which files should be read.

    Returns:
        data (List[STS]): Contains the imported data as STS objects.
    """
    # Check if path leads to a file or a folder
    if os.path.isfile(path):
        paths = [path]

    # otherwise, path points to a folder so,
    # get all paths in folder according to filter
    else:
        paths = glob.glob(os.path.join(path, f"*{filter}*"))

    # Check if any files were found
    if len(paths) == 0:
        raise ValueError(FILES_NOT_FOUND_ERROR)

    # Read files with appropriate import function
    spectra = read_spectra(paths, source)

    return spectra
