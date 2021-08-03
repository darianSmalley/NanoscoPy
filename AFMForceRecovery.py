import numpy as np
import math
from scipy.special import hyp2f1

################################################################
################ Define some helper functions ##################
################################################################

def lennard_jones(r , sigma , epsilon , LJ_form = 'standard'):
    """
    Calculate a Lennard-Jones potential.
    
    Inputs:
    r: numpy array. Contains distances between interacting particles in units of m.
    sigma: float or int. Distance at which potential energy is zero in units of m.
    epsilon: float or int. 'Dispersion energy' (or depth of potential well) for the potential in units of J.
    LJ_form: string. Specifies whether the user wants the 'standard' 6-12 L.J potential or the modified version.
    
    Outputs:
    The output is the calculated potential as a numpy array.
    """
    if LJ_form == 'standard':
        # Generate the standard '6-12' Lennard-Jones potential.
        return 4 * epsilon * ( np.divide(sigma , r) ** 12 - np.divide(sigma , r) ** 6 )
    elif LJ_form == 'modified':
        # Generate a modified version of the '6-12' Lennard-Jones potential.
        return - epsilon * ( 2 * np.divide(sigma , r) ** 6 - np.divide(sigma , r) ** 12 )

def Force_from_Potential(r , V):
    """
    Calculates the force from a given potential using the 'negative gradient' technique from Physics I.
    
    Inputs:
    r: numpy array. Distance from the potential's origin in units of m.
    V: numpy array. Potential values for corresponding distances in r.
    
    Outputs:
    The output is the force corresponding to the input potential in units of N. Length of output vector is one less than the input vectors.
    
    Note: This function uses the strict 'difference quotient' method of determining the derivative for consistency with other published numeric methods.
        Another option (shown below in comment form) would be to use Numpy's built-in gradient function. This would result in an output vector whose
        length is equal to that of the input vectors.
    """
    #return -np.gradient(V , r)
    return - np.diff(V) / np.diff(r)
    
def frequency_shift(z_ltp , f_0 , k , A , E_bond , sigma , potential_type = 'Lennard-Jones-Modified'):
    """
    Calculates simulated frequency shift data to model an FM-AFM experiment.
    
    Inputs:
    z_tlp: numpy array. Contains distances of closest approach of an AFM tip to a sample in units of m.
    f_0: float or int. Resonance frequency of unloaded cantilever. Should be in units of Hz.
    k: float or int. Effective spring constant of cantilever. Units of N/m.
    A: float or int. The oscillation amplitude of the AFM cantilever during experiment. Should be in units of m.
    E_bond: float or int. 'Dispersion energy' (or depth of potential well) for the potential in units of J.
    sigma: float or int. Distance at which potential energy is zero in units of m.
    potential_type: string. Specifies what type of potential the frequency shift data should be generated for.
    
    Outputs:
    The output is the simulated frequency shift in units of Hz as a numpy array.
    
    Note: For more information regarding the calculations performed by this function, see the following journal (as well as additional articles cited therein):
       J. Welker, E. Illek and F. Giessibl
       "Analysis of force-deconvolution methods in frequency-modulation atomic force microscopy"
       Beilstein Journal of Nanotechnology, 2012, 3, 238–248. 
       DOI: https://doi.org/10.3762/bjnano.3.27
    
    """
    if potential_type == 'Lennard-Jones-Modified':
        # Simulate the data based on the modified '6-12' Lennard-Jones potential.
        prefactor = - 12 * f_0 * E_bond / (k * A * sigma)
        argument = np.divide(- 2 * A , z_ltp)
        term1 = np.multiply( np.divide(sigma , z_ltp)**7 , hyp2f1(7 , 0.5 , 1 , argument) - hyp2f1(7 , 1.5 , 2 , argument) )
        term2 = np.multiply( np.divide(sigma , z_ltp)**13 , hyp2f1(13 , 0.5 , 1 , argument) - hyp2f1(13 , 1.5 , 2 , argument) )
        freq_shift = prefactor * (term1 - term2)
    elif potential_type == 'Morse':
        print('This feature has not been implemented yet.')
    return freq_shift

################################################################
############ Define Sader-Jarvis Method Function ###############
################################################################

def saderF(z , Delta_f , A , k , f_0):
    """
    Performs force recovery on frequency-modulated AFM data using the Sader-Jarvis method. 
    
    Inputs:
    z: numpy array. Contains tip height data. Should be in units of m.
    Delta_f: numpy array. Contains frequency shift data. Should be in units of Hz.
    A: float or int. The oscillation amplitude of the AFM cantilever during experiment. Should be in units of m.
    k: float or int. Effective spring constant of cantilever. Units of N/m.
    f_0: float or int. Resonance frequency of unloaded cantilever. Should be in units of Hz.
    
    Outputs:
    z: numpy array. Truncated version of the input height data. Included as convenience for plotting recovered force data. Has units of m.
    F: numpy array. Recovered force in units of N.
    
    Note: This function was adapted from MATLAB code written by the authors of the following journal article. The original MATLAB code can be found in the supplementary info section of the journal article on the publisher's webpage.
    
    Source Journal Article:
       J. Welker, E. Illek and F. Giessibl
       "Analysis of force-deconvolution methods in frequency-modulation atomic force microscopy"
       Beilstein Journal of Nanotechnology, 2012, 3, 238–248. 
       DOI: https://doi.org/10.3762/bjnano.3.27
    
    Theory Reference: 
       J. E. Sader and S. P. Jarvis
       "Accurate formulas for interaction force and energy in frequency modulation force spectroscopy"
       Applied Physics Letters, 84, 1801-1803 (2004).
       DOI: https://doi.org/10.1063/1.1667267

    Other Reference:
       J. E. Sader and S. P. Jarvis
       Mathematica notebook for implementation of formulas 
       http://www.ampc.ms.unimelb.edu.au/afm/bibliography.html#FMAFM. 
    """
    # Calculate spatial derivative of frequency shift. This code uses the difference quotient method of calculating the derivative rather than Numpy's gradient method.
    derivative = np.diff(Delta_f) / np.diff(z)
    
    # The input vectors need to have their length adjusted to match the derivative vector, since the difference quotient method outputs a derivative vector which is one element shorter than the inputs.
    z = z[:len(derivative)]
    Delta_f = Delta_f[:len(derivative)]

    # Calculate prefactor
    prefactor = 2 * k / f_0

    # Initialize a vector to store the recovered force values in.
    F_recovered = []
    
    # Calculate the recovered force for each data point. j is the index of the z value under consideration for any particular iteration of the loop.
    for j in range(0 , len(z) - 1): # Iterate over the whole (shortened) z vector except the last element. Skip the last element because this serves as the upper limit of the integral ("infinity"), which we won't use in a numeric integration.
        # Define t as the z range to be integrated over. Ranges from the lowest z value (i.e. the jth) under consideration, which is treated as the distance of closest approach.
        t = z[j+1:] # Skip the first z value (corresponding to z_j) to avoid the pole at t = z_j.

        # Pick out the portions of vectors whose z values correspond to those in t.
        Delta_f_j = Delta_f[j+1:]
        derivative_j = derivative[j+1:]

        # Calculate the integrand.
        g_j1 = (1 + np.divide(math.sqrt(A) , 8 * math.sqrt(math.pi) * np.sqrt(t - z[j]))) * Delta_f_j
        g_j2 = np.divide(A**(3/2) , math.sqrt(2) * np.sqrt(t - z[j])) * derivative_j
        g_j = g_j1 - g_j2

        # Perform numeric integration using the trapezoidal rule
        integral = np.trapz(g_j, t - z[j] )

        # Calculate correction factor term-by-term
        corr1 = Delta_f[j] * (z[j+1] - z[j])
        corr2 = 2 * math.sqrt(A) / ( 8 * math.sqrt(math.pi)) * Delta_f[j] * math.sqrt(z[j+1] - z[j])
        corr3 = 2 * A**(3/2) / math.sqrt(2) * math.sqrt(z[j+1] - z[j]) * (Delta_f[j+1] - Delta_f[j]) / (z[j+1] - z[j])

        # Calculate the recovered force for the jth data point.
        F_j = prefactor * (corr1 + corr2 + corr3 + integral)

        # Save calculations to vectors
        F_recovered.append(F_j)
    return z[:-1] , np.array(F_recovered)