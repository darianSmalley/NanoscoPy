import numpy as np
import spiepy
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ..utilities import progbar


def rescale(image):
    ''' Rescale images to 0-255 as type unit8 for use with open CV '''
    return ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')


def CLAHE(image):
    ''' Contrast Limited Adaptive Histogram Equalization '''
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def line_flatten(image):
    image = np.array(image)
    output = np.zeros(image.shape)
    for i, row in enumerate(image):
        output[i] = row - np.mean(row)

    output = output - np.mean(output)
    return output


def basic_flatten(image, poly=True):
    ''' 2nd order polynomial plane fitting, then line-by-line average offset '''
    im = spiepy.Im()
    im.data = image
    im, _ = spiepy.flatten_xy(im)
    if poly:
        im, _ = spiepy.flatten_poly_xy(im, deg=2)
    im = line_flatten(im.data)
    return im


def basic_correction(image, poly, equalize=True):
    ''' basic flatten, then rescale to [0,255] as uint8, then equalize, finaly smooth.'''
    im = basic_flatten(image, poly)
    im = rescale(im)
    if equalize:
        im = CLAHE(im)
    im = cv2.GaussianBlur(im, (3, 3), cv2.BORDER_DEFAULT)
    return im


def flatten(images):
    output = []
    n = len(images)
    for i, image in enumerate(images):
        try:
            flattened = basic_flatten(image)
            output.append(flattened)
            progbar(i+1, n, 10, 'Corrcting images...')

        except Exception as error:
            print(error)
            output.append(image)

    return output


def correct(images, terrace=False, poly=True, equalize=True):
    output = []
    n = len(images)
    for i, image in enumerate(images):
        try:
            corrected = basic_correction(image, poly, equalize)

            if terrace:
                corrected = terrace_level(corrected)

            output.append(corrected)
            progbar(i+1, n, 10, 'Corrcting images...Done' if i +
                    1 == n else 'Corrcting images...')

        except Exception as error:
            print(error)
            output.append(image)

    print('')
    return output


def subtract_poly1D(image, poly_order=2, mask=None, axis='x'):
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
    if axis in ['y', 'Y']:
        image = image.transpose()

    # Initialize array for background image
    background_image = np.zeros_like(image)

    # Generate an blank mask if no mask is provided
    if mask is None:
        mask = np.zeros_like(image)

    # Generate vectors containing pixel indices for x and y dimensions of the image.
    x_vect = np.arange(0, np.shape(image)[0])
    y_vect = np.arange(0, np.shape(image)[1])
    [X_grid, _] = np.meshgrid(x_vect, y_vect)

    # Remove masked points so that they are not used in fitting.
    x_points = np.ma.array(X_grid, mask=mask).ravel().compressed()
    z_points = np.ma.array(image, mask=mask).ravel().compressed()

    # Fit polynomial to data along the x axis of the image.
    coeffs = np.polynomial.polynomial.polyfit(x_points, z_points, poly_order)

    # Generate an array of the fitted polynomial for background subtraction
    # Vector containing all x points in the image, not just unmasked.
    x_sub = X_grid.ravel()
    background_vect = np.polynomial.polynomial.polyval(x_sub, coeffs)
    background_image = background_vect.reshape(np.shape(X_grid))

    # Perform background subtraction of image.
    image_subtracted = image - background_image

    # Transpose images back to original orientation if y axis is selected for leveling. This is unecessary if x is selected.
    if axis in ['y', 'Y']:
        image_subtracted = image_subtracted.transpose()
        background_image = background_image.transpose()

    return image_subtracted, background_image


def subtract_poly1D_line(image, poly_order=2, mask=None, axis='x'):
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
    if axis in ['y', 'Y']:
        image = image.transpose()

    # Initialize array for background image
    background_image = np.zeros_like(image)

    # Generate an empty mask if no mask is provided
    if mask is None:
        mask = np.zeros_like(image)

    # Generate vectors containing pixel indices for x and y dimensions of the image.
    x_vect = np.arange(0, np.shape(image)[0])
    y_vect = np.arange(0, np.shape(image)[1])
    [X_grid, _] = np.meshgrid(x_vect, y_vect)

    # Create masked arrays for processing
    X_image = np.ma.array(X_grid, mask=mask)
    Z_image = np.ma.array(image, mask=mask)

    for row_num, [X_row, Z_row] in enumerate(zip(X_image, Z_image)):
        # Fit a polynomial to the unmasked data from each row of the image individually
        coeffs = np.polynomial.polynomial.polyfit(
            X_row.compressed(), Z_row.compressed(), poly_order)
        # Calculate polynomial background for all pixels of the image
        background_image[row_num] = np.polynomial.polynomial.polyval(
            X_grid[row_num], coeffs)

    # Subtract polynomial background from image
    image_subtracted = image - background_image

    # Transpose images back to original orientation if y axis is selected for leveling. This is unecessary if x is selected.
    if axis in ['y', 'Y']:
        image_subtracted = image_subtracted.transpose()
        background_image = background_image.transpose()

    return image_subtracted, background_image


def create_mask(image, mask_method='mask_by_mean'):
    """
    Creates a mask to exclude certain parts of an image. This can be useful for excluding troublesome sections of an image during flattening.

    Inputs:
        image: Numpy array. Contains the image to be masked.

    Outputs:
        mask: Numpy array. Contains a mask for an image. Pixels which are masked will be excluded by SPIEPy's levelling functions.
    """
    # Convert Numpy array image to SPIEPy image format.
    im = spiepy.Im()
    im.data = image

    # Create an image mask using one of SPIEPy's built-in methods.
    if mask_method == 'mean':
        # Use this if this if there is contamination, but no atomic resolution
        mask = spiepy.mask_by_mean(im)
    elif mask_method == 'peak-trough':
        # Use this if there is a fair amount of contamination in the imge, but also atomic resolution
        mask, _ = spiepy.mask_by_troughs_and_peaks(im)
    elif mask_method == 'step':
        # Use this if there are step edges in the image
        mask = spiepy.locate_steps(im, 4)
    else:
        print('Unknown masking type.')
        mask = np.zeros_like(image)
    return mask


def plot_masked_image(image, mask):
    """
    Shows an image with a pixel mask overlaid. If the mask was generated using SPIEPy functionality, the masked pixels will correspond to those which fail to meet some criteria.

    Inputs:
        image: Numpy array. Contains the image to be masked.
        mask: Numpy array. Contains a mask for an image. Pixels which are masked will be excluded by SPIEPy's levelling functions.
    """
    # Make a masked array, to visualize the mask superimposed over the image
    masked_image = np.ma.array(image, mask=mask)
    palette = spiepy.NANOMAP
    palette.set_bad('#00ff00', 1.0)
    # Show the masked image
    plt.imshow(masked_image, cmap=spiepy.NANOMAP, origin='lower')


def plot_mask_comparison(im_unleveled, mask, im_leveled, titles=['Masked Image', 'Leveled Image']):
    """
    Creates a figure showing both the original image and a masked version of the image. 

    Inputs:
        im_unleveled: Numpy array. Contains the image to be masked.
        mask: Numpy array. Contains the pixel mask.
        im_leveled: Numpy array. Contains the image after levelling has been performed.
        titles: list of strings. Should have exactly two elements, corresponding to the desired titles for the images.

    Outputs:
        fig: Matplotlib Figure object. Contains information about the compound figure.
        ax1, ax2: Matplotlib Axis objects. Contains info about the images for the compound figure.
    """
    # Generate masked image of unleveled image
    masked_image = np.ma.array(im_unleveled.data, mask=mask)
    palette = spiepy.NANOMAP
    palette.set_bad('#00ff00', 1.0)

    # Generate a subplot figure for plotting.
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the masked image on the subplot figure
    ax1.imshow(masked_image, cmap=spiepy.NANOMAP, origin='lower')
    ax1.set_title(titles[0])
    # Plot the leveled image on the subplot figure
    ax2.imshow(im_leveled.data, cmap=spiepy.NANOMAP, origin='lower')
    ax2.set_title(titles[1])

    return fig, (ax1, ax2)


def terrace_level(image):
    """
    Attempts to level an image by finding terraces in the image and calculating the needed transform to make the terraces level on the average.

    Inputs:
        image: Numpy array. Contains the image to be masked.

    Outputs:
        Numpy array. Levelled image.
    """
    # Convert Numpy array image to SPIEPy image format.
    im = spiepy.Im()
    im.data = image

    # Pre-flatten the image using a plane fit to remove large-scale tilt in the image.
    im_preflat, _ = spiepy.flatten_xy(im)

    # Locate the step
    mask_step = create_mask(im_preflat.data, mask_method='step')

    # Flatten the image, taking the step into account
    im_leveled_step, _ = spiepy.flatten_xy(im, mask_step)

    # Return the flattened image as a Numpy Array.
    return im_leveled_step.data
