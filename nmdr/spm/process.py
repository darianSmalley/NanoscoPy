import numpy as np
import spiepy
from pathlib import Path
import cv2

def line_flatten(image):
    image = np.array(image)
    output = np.zeros(image.shape)
    for i, row in enumerate(image):
        mean = np.mean(row)
        output[i] = row - mean
    
    return output

def basic_flatten(image):
    im = spiepy.Im()
    im.data = image
    im, _ = spiepy.flatten_xy(im)
    im, _ = spiepy.flatten_poly_xy(im, deg=2)
    im = line_flatten(im.data)
    im = cv2.GaussianBlur(im,(3,3), cv2.BORDER_DEFAULT)
    return im

def flatten(images):
    output = []
    for image in images:
        try:
            flattened = basic_flatten(image)
            output.append(flattened)

        except Exception as error:
            print(error)
            output.append(image)
    
    return output

def subtract_poly1D(image , poly_order = 2, mask = None , axis = 'x'):
    """
    Performs polynomial background subtraction on a whole image along one axis. The whole image is fitted with a single polynomial.

    Inputs:
        image: numpy array. Image to undergo polynomial background subtraction (usually height map image from STM or AFM).
        poly_order: int. The order of the polynomial to be used in background subtraction.
        mask: numpy array. Mask specifying which pixels are to be ignored during background subtraction (pixels which are True are ignored, False are kept).
        axis: string. Specifies the axis of the iamge along which the polynomial should be fitted. Options are 'x', 'X' , 'y' or 'Y'.
    
    Outputs:
        image_subtracted: numpy array. The resulting image after performing background subtraction on the inputted image.
        background_image: numpy array. The image of the polynomial background which was subtracted from the inputted image.
    """
    # Transpose image if y direction is selected for leveling. This puts the y axis of the original image along the x axis during processing.
    if axis in ['y' , 'Y']:
        image = image.transpose()

    # Initialize array for background image
    background_image = np.zeros_like(image)

    # Generate an blank mask if no mask is provided
    if mask is None:
        mask = np.zeros_like(image)
    
    # Generate vectors containing pixel indices for x and y dimensions of the image.
    x_vect = np.arange(0 , np.shape(image)[0])
    y_vect = np.arange(0 , np.shape(image)[1])
    [X_grid , _] = np.meshgrid(x_vect , y_vect)

    # Remove masked points so that they are not used in fitting.
    x_points = np.ma.array(X_grid , mask = mask).ravel().compressed()
    z_points = np.ma.array(image , mask = mask).ravel().compressed()

    # Fit polynomial to data along the x axis of the image.
    coeffs = np.polynomial.polynomial.polyfit(x_points , z_points , poly_order)

    # Generate an array of the fitted polynomial for background subtraction
    x_sub = X_grid.ravel() # Vector containing all x points in the image, not just unmasked.
    background_vect = np.polynomial.polynomial.polyval(x_sub , coeffs)
    background_image = background_vect.reshape(np.shape(X_grid))

    # Perform background subtraction of image.
    image_subtracted = image - background_image

    # Transpose images back to original orientation if y axis is selected for leveling. This is unecessary if x is selected.
    if axis in ['y' , 'Y']:
        image_subtracted = image_subtracted.transpose()
        background_image = background_image.transpose()

    return image_subtracted , background_image

def subtract_poly1D_line(image , poly_order = 2, mask = None , axis = 'x'):
    """
    Performs polynomial background subtraction of an image in a line-by-line fashion. Each line is fitted and background-subtracted separately.
        
    Inputs:
        image: numpy array. Image to undergo polynomial background subtraction (usually height map image from STM or AFM).
        poly_order: int. The order of the polynomial to be used in background subtraction.
        mask: numpy array. Mask specifying which pixels are to be ignored during background subtraction (pixels which are True are ignored, False are kept).
        axis: string. Specifies the axis of the iamge along which the polynomial should be fitted. Options are 'x', 'X' , 'y' or 'Y'.
    
    Outputs:
        image_subtracted: numpy array. The resulting image after performing background subtraction on the inputted image.
        background_image: numpy array. The image of the polynomial background which was subtracted from the inputted image.
    """
    # Transpose image if y direction is selected for leveling. This puts the y axis of the original image along the x axis during processing.
    if axis in ['y' , 'Y']:
        image = image.transpose()

    # Initialize array for background image
    background_image = np.zeros_like(image)

    # Generate an empty mask if no mask is provided
    if mask is None:
        mask = np.zeros_like(image)
    
    # Generate vectors containing pixel indices for x and y dimensions of the image.
    x_vect = np.arange(0 , np.shape(image)[0])
    y_vect = np.arange(0 , np.shape(image)[1])
    [X_grid , _] = np.meshgrid(x_vect , y_vect)

    # Create masked arrays for processing
    X_image = np.ma.array(X_grid , mask = mask)
    Z_image =  np.ma.array(image , mask = mask)

    for row_num , [X_row , Z_row] in enumerate(zip(X_image , Z_image)):
        # Fit a polynomial to the unmasked data from each row of the image individually
        coeffs = np.polynomial.polynomial.polyfit(X_row.compressed() , Z_row.compressed() , poly_order)
        # Calculate polynomial background for all pixels of the image
        background_image[row_num] = np.polynomial.polynomial.polyval(X_grid[row_num] , coeffs)
    
    # Subtract polynomial background from image
    image_subtracted = image - background_image
    
    # Transpose images back to original orientation if y axis is selected for leveling. This is unecessary if x is selected.
    if axis in ['y' , 'Y']:
        image_subtracted = image_subtracted.transpose()
        background_image = background_image.transpose()

    return image_subtracted , background_image