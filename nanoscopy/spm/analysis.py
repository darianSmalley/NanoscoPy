import numpy as np
import math
from lmfit import Model

def calculate_HHCF(image , image_width):
    """
     Calculates the height-height correlation function for an image. The HHCF is calculated along the rows of the image.

     An Open Access reference for calculating HHCF from an image:
        "Height-Height Correlation Function to Determine Grain Size in Iron Phthalocyanine Thin Films", Gredig, T; Silverstein, E; Byrne, M. Journal of Physics:Conference Series. (2013). DOI: 10.1088/1742-6596/417/1/012069

     Inputs:
        image: numpy array. Contains the height image.
        image_width: float. Specifies the physical width corresponding to the image frame expressed in meters. (E.g. a 200 nm wide scan frame should be inputted as 1e-9.)
    
    Outputs:
        distances: numpy array. Contains the distances values from the reference pixel corresponding to each value in the HHCF array. This is the X data for plotting in a HHCF plot.
        HHCF: numpy array. Contains the calculated height-height correlation function data.
    """
    # Initialize parameters and an array for data storage.
    num_columns = np.shape(image)[0]
    HHCF = np.zeros(num_columns)

    # Perform calculation of HHCF. This works by finding the difference in value between pairs of pixels separated by a set number of pixels (corresponding to a physical distance).
    # This is a semi-vectorized implementation of the calculation. The idea is to effectively have two copies of the image stacked atop one another. Then the top image is shifted to the left relative to the bottom image by some number of pixels (corresponding to a certain physical difference).
    # Then the two copies of the image are 'cropped' to contain only the portions of the images which overlap with their shifted counterparts, discarding any pixels which do not have a pair.
    # Loop over all allowable (non-zero) pixel-wise separation distances.
    for separation_distance in range(1,num_columns):
        # Slice the image to generate the 'bottom' copy of the image as described above, discarding pixels on the right side, since they won't have pairs in the 'top' copy of the image.
        array1 = image[: , 0:-separation_distance]
        # Slice the image to generate the 'top' copy of the image as described above, discarding pixels on the left side, since they won't have pairs in the 'bottom' copy of the image.
        array2 = image[: , separation_distance:]
        # Calculate the HHCF value for each pixel pair in the overlapping regions of the 'image stack', then take the average to represent the HHCF for the current pixel-wise separation distance.
        HHCF[separation_distance] = np.average(np.square(array2-array1))
    
    # Determing the physical length corresponding to a single pixel.
    pixel_width = image_width/(num_columns - 1)
    
    # Generate an array of physical separation distances corresponding to the pixel-wise separation distances used to calculate the HHCF.
    distances = np.arange(num_columns) * pixel_width
    
    return [distances , HHCF]

def HHCF(r, sigma, xi , alpha):
    """
    Model for height-height correlation function. This model is suitable for fitting when data is present for both above and below the correlation length.

    Inputs:
        r: Numpy array. This contains the distance data which would be used for the x coordinate on a HHCF plot.
        sigma: float. This is related to the RMS roughness of the sample.
        xi: float. This is the correlation length of the sample.
        alpha: float. This is the Hurst parameter.
    
    Outputs:
        The output of this function is the height-height correlation function for the given distances and parameters.
    """
    return 2 * sigma**2 * (1 - np.exp(- (r/xi)**(2 * alpha)))

def fit_HHCF(HHCF_data , parameter_guesses = None):
    """
    Fit an analytic model for height-height correlation function to experimental HHCF data. This function assumes the HHCF data includes length scales both above and below the correlation length.

    The analytic model is mentioned in the following Open Source article:
        "Height-Height Correlation Function to Determine Grain Size in Iron Phthalocyanine Thin Films", Gredig, T; Silverstein, E; Byrne, M. Journal of Physics:Conference Series. (2013). DOI: 10.1088/1742-6596/417/1/012069
    
    Inputs:
        HHCF_data: Numpy array. Contains two columns, with the first column being the distance data and the second being the HHCF calculated from an experimental image.
        parameter_guesses Dict. This is an optional input containing guesses for the model parameters. If it is provided, it must have key:value pairs for the following parameters: sigma, xi, alpha.
    """
    # Generate initial guesses of model parameters for fitting if none are provided.
    if parameter_guesses == None:
        parameter_guesses = {'sigma':math.sqrt(np.max(HHCF_data[1])/2)}
        index_of_max = np.argmax(HHCF_data[1])
        parameter_guesses['xi'] = HHCF_data[0][int(index_of_max/4)]
        parameter_guesses['alpha'] = 0.5
    
    # Create a model object to fit to the HHCF data and assign initial guesses for the fitting parameters.
    fitting_model = Model(HHCF)
    params = fitting_model.make_params(sigma = parameter_guesses['sigma'], xi = parameter_guesses['xi'], alpha = parameter_guesses['alpha'])
    
    # Try to fit the model to the data.
    try:
        result = fitting_model.fit(HHCF_data[1], params, r = HHCF_data[0])
        return result
    except:
        print('There was an issue processing the data.')
    
def HHCF_short_range(r, A, alpha):
    """
    Model for height-height correlation function. This model is suitable for fitting when data is present only below the correlation length.

    Inputs:
        r: Numpy array. This contains the distance data which would be used for the x coordinate on a HHCF plot.
        A: float. Scaling constant used in the fitting process.
        alpha: float. This is the Hurst parameter.

    Outputs:
        The output of this function is the height-height correlation function for the given distances and parameters.
    """
    return A * np.power(r , 2 * alpha)

    ####### The function below still needs work!
def fit_HHCF_short_range(HHCF_data , parameter_guesses = None):
    """
    Fit an analytic model for height-height correlation function to experimental HHCF data. This function assumes the HHCF data includes only length scales below the correlation length.

    The analytic model is mentioned in the following Open Source article:
        "Height-Height Correlation Function to Determine Grain Size in Iron Phthalocyanine Thin Films", Gredig, T; Silverstein, E; Byrne, M. Journal of Physics:Conference Series. (2013). DOI: 10.1088/1742-6596/417/1/012069
    
    Inputs:
        HHCF_data: Numpy array. Contains two columns, with the first column being the distance data and the second being the HHCF calculated from an experimental image.
        parameter_guesses: Dict. This is an optional input containing guesses for the model parameters. If it is provided, it must have key:value pairs for the following parameters: A, alpha.
    """
    # Generate initial guesses of model parameters for fitting if none are provided.
    if parameter_guesses == None:
        # Guess the parameter values by assuming that the data will be linear on a log-log plot and using a linear fit.
        mid = int( len(HHCF_data) / 2)
        # Calculate the log of two data points near the middle of the data array.
        x = np.log( HHCF_data[0][mid : mid + 5])
        y = np.log( HHCF_data[1][mid : mid + 5] )
        # Perform a 'linear regression' according to: log(HHCF) = log(A) + 2 * alpha * log(r)
        alpha_guess = 0.5 * (y[1] - y[0]) / (x[1] - x[0]) # slope
        A_guess = np.exp((y[0]*x[1] - y[1]*x[0]) / (x[1] - x[0])) # y-intercept
        # Store parameter guesses in a dictionary.
        parameter_guesses = {'A': A_guess , 'alpha':alpha_guess}
    
    # Create a model object to fit to the HHCF data and assign initial guesses for the fitting parameters.
    fitting_model = Model(HHCF_short_range)
    params = fitting_model.make_params(A = parameter_guesses['A'], alpha = parameter_guesses['alpha'])
    
    # Try to fit the model to the data.
    try:
        result = fitting_model.fit(HHCF_data[1], params, r = HHCF_data[0])
        return result
    except:
        print('There was an issue processing the data.')

def make_histogram(image, num_bins = 100):
    """
    Bin a collection of numeric data for use in histogram plotting or analysis.

    Inputs:
        image: Numpy array. Contains the data to be binned.
        num_bins: int. Specifies the number of bins to use.
    
    Outputs:
        bin_centers: Float. The center coordinate for the histogram bins.
        counts: Int. The number of data points contained in each bin.
        bin_width: Float. The width of the bins.
    """
    # Group data into desired number of bins.
    counts , bin_edges = np.histogram(image , bins = num_bins)

    # calculate the center values for the bins
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # calculate the width of the bins.
    bin_width = (bin_edges[-1]-bin_edges[0])/len(counts)
    return bin_centers, counts, bin_width
    