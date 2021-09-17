from scipy.signal import find_peaks, peak_prominences, savgol_filter
from scipy.special import erfc
from lmfit.models import GaussianModel , LorentzianModel, VoigtModel , ConstantModel
import numpy as np
from math import pi, sqrt, log , exp

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

class Spectrum:
    def __init__(self,input_data,material=None):    
        """
        input_data: (numpy array) with the leftmost column being the x data.
        material: (string) default is None
        """
        self.data_x = input_data[:,0]
        self.data_y = input_data[:,1]
        self.material = material
        self.model = None
        self.results = None
    
    def filter_data(self,x_min,x_max):
        """
        x_min: (numeric) lower endpoint of desired x data range.
        x_max: (numeric) upper endpoint of desired x data range.
        """
        keep_elements = (self.data_x >= x_min) & (self.data_x <= x_max)
        self.data_x = self.data_x[keep_elements]
        self.data_y = self.data_y[keep_elements]
    
    def locate_peaks(self,filter_window_length=9,filter_polyorder=1,finder_prominence=20, wlen=30, width=5):
        """
        Search through the data and find peaks.
        """
        # Smooth data before peak finding (to reduce the influence of noise)
        Intensity_denoised = savgol_filter(self.data_y, filter_window_length, filter_polyorder)
        # Find the peaks in the denoised intensity data.
        peak_indices, peak_properties = find_peaks(Intensity_denoised, prominence=finder_prominence, wlen=wlen, width=width)
        return peak_indices , peak_properties

    def generate_model_parameters(self,peak_indices,peak_properties,peak_type='Lorentzian'):
        """
        Use this to generate the model parameters for a set of peaks automatically found using the "loacate_peaks" method.
        """
        generated_model = []
        for i in range(len(peak_indices)):
            new_peak = {'name':'peak_'+str(i),
                        'type': peak_type,
                        'center': {'value':self.data_x[peak_indices[i]]},
                        'height': {'value':self.data_y[peak_indices[i]]},
                        'FWHM': {'value':peak_properties['widths'][i]}
                        }
            generated_model.append(new_peak)
        return generated_model

    def define_peak(self,peak_parameters,peak_num):
        # Define the peak prefix. [To do: Add option to auto-generate name.]
        peak_prefix = peak_parameters['name'] + '_' #'p' + str(peak_num) + '_'
        
        # Define the peak shape. [To do: add functionality to select other peaks.]
        if peak_parameters['type'] in ['Lorentzian', 'lorentzian']:
            peak_model = LorentzianModel(prefix = peak_prefix)
        elif peak_parameters['type'] in ['Gaussian', 'gaussian']:
            peak_model = GaussianModel(prefix = peak_prefix)
        elif peak_parameters['type'] in ['Voigt', 'voigt']:
            peak_model = VoigtModel(prefix = peak_prefix)
        
        amplitude , sigma = convert_parameters(peak_parameters)

        # Set an initial guess for the peak center position.
        center_value = peak_parameters['center']['value']
        peak_model.set_param_hint('center', value = center_value, min= 0.9*center_value, max=1.1*center_value )

        # Set initial guess for peak height.
        #peak_model.set_param_hint('amplitude', value=peak_parameters['height']['value'])
        peak_model.set_param_hint('amplitude', value=amplitude, min=amplitude/5,max=amplitude*5)

        # Set initial guess for peak width. [To do: add function to convert between FWHM and lmfit's sigma.]
        #peak_model.set_param_hint('sigma', value = peak_parameters['FWHM']['value']/2)
        peak_model.set_param_hint('sigma', value = sigma, min = sigma/5, max=sigma*5)

        return peak_model
    
    def make_composite_model(self,model_parameters):
        # Initialize the model with a background model.
        background = ConstantModel(prefix='back_')
        background.set_param_hint('c', value=np.min(self.data_y), min=1e-6)
        self.model = background

        # Add peak models.
        peak_num = 0
        for peak_parameters in model_parameters:
            self.model = self.model + self.define_peak(peak_parameters,peak_num)
            peak_num = peak_num + 1
    
    def fit_spectrum(self):
        params = self.model.make_params()
        self.results = self.model.fit(self.data_y, params, x=self.data_x)
    
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