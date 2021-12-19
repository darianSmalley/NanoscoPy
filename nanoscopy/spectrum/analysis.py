from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.special import erfc
from lmfit.models import GaussianModel , LorentzianModel, VoigtModel , ConstantModel
import numpy as np
from math import pi, sqrt, log , exp
from .peak_fit import PeakFit

# To do list:
#   - Make it so that pre-defined models also use the height/FWHM-> amplitude/sigma conversions.
#   - Add documentation to this file.
#   - Make a 'tutorial' jupyter notebook demonstrating functionality and how to use it.
#   - Add additional models (for MoS2, WS2, maybe some PL models, etc.)
#   - Make a function which will automatically handle 'custom' or 'None' materials, automatically calling the locate_peaks, etc. functions to create a model
#   - Make a function which will append results to a dataframe and/or output to a file
#   - Move the file loader/conditioner to a file in a separate module and include a 'translator' file for the Renishaw files.
#   - Implement the height/FWHM -> amplitude/sigma conversions for Voigt type peaks
#   - Add a check to see if there is a fitting model define before running the fitting algorithm
#   - Add functionality to let the user choose what kind of background to use (if any).
#   - Add functionality to let the user normalize the data (norm. to max value, area under curve, etc.)
#   - Add functionality to let the user specify bounds for the fitting parameters.
#   - Add functionality to let the user save/load models from a file.
#   - Add functionality to make a plot showing the individual peaks overlaid on the original data.

def convert_parameters(peak_parameters):
        height = peak_parameters['height']['value']
        FWHM = peak_parameters['FWHM']['value']
        
        if peak_parameters['type'] in ['Lorentzian', 'lorentzian']:
            sigma = FWHM/2
            amplitude = height * sigma * pi
        elif peak_parameters['type'] in ['Gaussian', 'gaussian']:
            sigma = FWHM / (2 * sqrt(2 * log(2) )) # The denominator is approximately equal to 2.35482004503.
            amplitude = height * sigma * sqrt(2 * pi)
        elif peak_parameters['type'] in ['Voigt', 'voigt']:
            sigma = FWHM / 3.6013 # Approximation from lmfit documentation for VoigtModel
            
            # Define variables needed to calculate parameters for Voigt peak
            gamma = sigma # It is commonly assumed that the parameter gamma is equal to sigma
            z = 1j * gamma / (sigma * sqrt(2)) # Note: j is the imaginary unit (sqrt(-1))
            w = np.exp(-z**2) * erfc(-1j * z)
            amplitude = height * sigma * sqrt(2 * pi) / w.real
        else:
            print('Function not implemented.')
        return amplitude , sigma

model_graphene = [
    {'name':'pk_D',
    'type': 'Lorentzian',
    'center': {'value':1350},
    'height': {'value':30},
    'FWHM': {'value':38} 
    } ,
    {'name':'pk_G',
    'type': 'Lorentzian',
    'center': {'value':1587},
    'height': {'value':700},
    'FWHM': {'value':12} 
    } ,
    {'name':'pk_DplusDprime',
    'type': 'Gaussian',
    'center': {'value':2457},
    'height': {'value':80},
    'FWHM': {'value':40} 
    } ,
    {'name':'pk_2D',
    'type': 'Voigt',
    'center': {'value':2690},
    'height': {'value':1800},
    'FWHM': {'value':30} 
    }
    ]