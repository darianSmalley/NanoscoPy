# This cell needs to be cleaned up some, as it's likely that not all of the imported packages will be
# used in the final code.
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Below are packages which have been used uring various stages of development,
# but may or may not be used in the final version (and should be cleaned up to remove any
# extraneous packages at the end.
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import savgol_filter
import numpy.polynomial.polynomial as poly
import glob
import os
import sys
from lmfit import Model, models, Parameters, report_fit
from lmfit.models import LinearModel, ConstantModel, VoigtModel, LorentzianModel, GaussianModel
from lmfit.lineshapes import gaussian, lorentzian

def read_raman_data():
    """Read raman data, sort, reindex and drop old indicies. Returns formatted dataframe."""
    # Specify file location and name
    filename = '2019012-18_Graphene_JTDS_'
    extension = '.txt'
    subfolder = 'Raman_Data'
    directory = 'C:\\Users\\Jesse\\OneDrive\\Python-Stuff\\Research'
    designation = 'Top'
    path = directory+'\\'+subfolder+'\\'+filename+designation+'_'+extension

    # Load data into a dataframe and assign more descriptive column labels
    df = pd.read_csv(path, sep='\t', engine='python',
                     names=['Raman Shift', 'Intensity'])

    # Sort dataframe in ascending order
    df = df.sort_values('Raman Shift')

    return df


def filter_data(df, shiftMin, shiftMax):
    """Selects the data within the desired range: ShiftMin <= v <= ShiftMax"""
    df_filtered = df[(df['Raman Shift'] > shiftMin) &
                     (df['Raman Shift'] < shiftMax)]
    return df_filtered


def reindex_data(df):
    # Reset the indexing of the dataframe to allow for normal slicing with the reversed order
    df.reset_index(inplace=True)

    # Drop previous column of indcies which we will not use
    df = df.drop(columns=['index'])

    return df


def fit_data(df):
    """Sets up the fitting models for the individual peaks. These individual models will be combined into a composite model used to fit to the whole spectrum."""
    # Perform smoothing of the data as a means of denoising.
    # Filter window length and fitting order may need to be optimized for different datasets.
    Intensity_denoised = savgol_filter(df['Intensity'], 9, 1)

    # Plot the resulting, smoothed data
    _, ax1 = plt.subplots()
    plt.plot(df['Raman Shift'], df['Intensity'], label='Raw Data')
    plt.plot(df['Raman Shift'], Intensity_denoised,
             'r-', label='Smoothed Data')
    plt.legend()

    # Find the approximate peak locations. peak_properties contains estimates of the widths of the peaks.
    # The prominence, wlen and peak width parameters may need to be optimized for different datasets.
    peak_indices, peak_properties = find_peaks(
        Intensity_denoised, prominence=20, wlen=30, width=5)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    plt.plot(df['Raman Shift'], Intensity_denoised, label='Smoothed Data')
    ax1.scatter(df['Raman Shift'][peak_indices], df['Intensity']
                [peak_indices], s=40, color='r', label="Peaks")
    plt.legend()

    # Initialize the model for the background with a prefix for future reference. Typically this will either be ConstantModel or LinearModel.
    background = ConstantModel(prefix='back_')

    # Give the fitting routine an initial guess of parameter for this model.
    background.set_param_hint('c', value=50, min=0, max=1000)

    # Set the background to be the first sub-model in the full composite model to describe the whole spectrum.
    composite_model = background

    # Initialize one individual model for each detected peak.
    for n in range(0, len(peak_indices)):
        # Define the model type to be used for the peak.
        # Note: This code assumes that all the peaks are to be fitted to the same type of peak (a lorentzian, in this case),
        # but it may be the case that different peaks may be better fitted by different peak shapes (e.g. gaussian, lorenztian or voigt)
        # due to presence or absence of broadening (due to thermal effects, defects, etc.)
        model = LorentzianModel(prefix='p' + str(n) + '_')

        # Determine and set the approximate center position of the peak (from the earlier peak-finding)
        peak_center = df['Raman Shift'][peak_indices[n]]
        model.set_param_hint('center', value=peak_center,
                             min=peak_center - 15, max=peak_center + 15)

        # Determine and set the approximate peak width (actual name for this depends on the model chosen) from the previous peak-fitting.
        peak_width = peak_properties['widths'][n]
        model.set_param_hint('sigma', value=peak_width,
                             min=0.5 * peak_width, max=2 * peak_width)

        # Determine and set the approximate amplitude of the peak. Amplitude is related to the max height of the peak, but is not identical too it.
        peak_amplitude = df['Intensity'][peak_indices[n]] * \
            math.pi * peak_width
        model.set_param_hint('amplitude', value=peak_amplitude,
                             min=0.75 * peak_amplitude, max=1.25 * peak_amplitude)

        # Add the individual model to the composite model.
        composite_model = composite_model + model
        params = composite_model.make_params()
        output = composite_model.fit(
            df['Intensity'], params, x=df['Raman Shift'])

    return output

df = read_raman_data()
df = filter_data(df, shiftMin=1200, shiftMax=3400)
df = reindex_data(df)
Intensity_denoised = savgol_filter(df['Intensity'], 9, 1)
#model = fit_data(df)
#fig4, ax4 = model.plot(data_kws={'markersize': 1})
#plt.ylim(0, 2000)
#print(model.fit_report())