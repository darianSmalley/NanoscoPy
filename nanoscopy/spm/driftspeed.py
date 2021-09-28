import cv2
import numpy as np

class CoordStorage:
    """
    Stores coordinates of point selected by clicking the left mouse button over an OpenCV imshow window.

    There are no direct outputs, as the coordinates are saved as a class property to be accessed as such. 
    """
    def __init__(self):
        self.points = []
    # The following function is used to tell OpenCV what to do when it detects a left mouse button click on the image.
    def select_point(self , event , x , y , flags , param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save image coordinates of the pixel which was clicked.
            self.points.append((x,y))
            # Notify user and close image.
            print('Click detected. Closing window.')
            # Close the imshow window (and all other OpenCV windows)
            cv2.destroyAllWindows()

def numpy_to_im(im):
    """
    Rescales an image of floats to have pixel intensity values in the range of a standard image (0 to 255).

    Input:
        im: Numpy array of floats. The image to be converted.

    Output:
        im_out: Numpy array of floats. Pixel intensities remapped to be between 0 and 255. Suitable for display in OpenCV's imshow function.
    """
    # Rescale an array to have values between 0 and 255.
    # Calculate highest value in array (to be mapped to 255)
    high = np.max(im)
    # Calculate lowest value in array (to be mapped to 0)
    low = np.min(im)
    # Calculate total span of values in array (should fit in the range of 0-255)
    difference = high - low
    # Offset lowest value to be 0 and rescale to fit in range of 255.
    im_out = (im - low ) * (255 / difference)
    return im_out

def get_imcoords(im):
    """
    Displays an image in a new window, prompts the user to click a point of interest, and outputs the image coordinates of the clicked point.
    
    Input:
        im: numpy array of floats or ints. Should contain the image to be displayed.

    Output:
        coordinates: tuple. (x,y) Image coordinate pair for the clicked point. Is in units of pixels. The origin of the image is in the top left corner.
    """
    # Initialize persistent storage for mouse click coordinates.
    coords = CoordStorage()
    # Check if image is outside of standard image pixel values.
    im = numpy_to_im(im)
    # Display first image so that the user can click on it. 
    #cv2.imshow('Image' , im)
    cv2.imshow('Image' , np.array(im, dtype = np.uint8 ))
    # Tell OpenCV to wait for a mouse click on the image and then tell it what to do with the click.
    cv2.setMouseCallback('Image', coords.select_point)
    # Wait 10 seconds for a mouse click
    cv2.waitKey(10000)
    # Close the window if no mouse click has been detected.
    cv2.destroyAllWindows()
    return coords.points[0]

def get_coords(data , widths):
    """
    Used to prompt the user to select a point on an image and convert the selected coordinates into physical coordinates (for an SPM image)

    Inputs:
        data: numpy array of floats. Contains the SPM image data (not the SPMImage class object) to be displayed.
        widths: numpy array of floats. Contains the physical widths corresponding to the SPM scan frame width. Should be in units of m.
    
    Outputs:
        x: float. The x coordinate of the selected point in units of m. Assumes that the origin is at the bottom left corner of the image.
        y: float. The y coordinate of the selected point in units of m. Assumes that the origin is at the bottom left corner of the image.
    """
    # Determine the number of pixels along the horizontal and vertical axes of the image.
    [x_size , y_size] = np.shape(data)
    # Assign variable names to the physical widths corresponding to the SPM scan frame used to acquire the image.
    x_width , y_width = widths[0] , widths[1]
    # Prompt the user to select a point on the image and return the image coordinates (units of pixels) of the selected point.
    coords = get_imcoords(data)
    # Convert image coordinates to physical coordinates. The selected points will have their origin in the upper left corner of the image.
    x = coords[0] / x_size * x_width 
    y = coords[1] / y_size * y_width
    # Convert the y coordinate so that the origin is in the lower left corner of the image.
    y = y_width - y
    return x , y

def calc_driftspeed(spm_im1 , spm_im2):
    """
    Calculates the drift speed observed between two SPM images taken without moving the scan frame between acquiring the two images.

    Inputs:
        spm_im1: SPMImage class object (from Nanoscopy package). Should contain a 'Z' channel image.
        spm_im2: SPMImage class object (from Nanoscopy package). Should contain a 'Z' channel image.
    
    Outputs:
        v_x: float. The calculated drift speed along the horizontal direction of the image coordinate system in units of m/s.
        v_y: float. The calculated drift speed along the vertical direction of the image coordinate system in units of m/s.        
    """
    # Get timestamp data and find difference between the acquisition times
    t1 = spm_im1.params['date_time']
    t2 = spm_im2.params['date_time']
    time_diff = t2 - t1
    # Prompt user for the point on the first image.
    print('Please use the LMB to select a point on the image.')
    x1 , y1 = get_coords(spm_im1['Z'][0] , [spm_im1.params['width'] , spm_im1.params['height']])
    # Prompt the user for the point on the second image.
    print('Please select the corresponding point on this image.')
    x2 , y2 = get_coords(spm_im2['Z'][0] , [spm_im2.params['width'] , spm_im2.params['height']])
    # Calculate drift speeds
    v_x = (x2 - x1) / time_diff.total_seconds()
    v_y = (y2 - y1) / time_diff.total_seconds()
    return v_x , v_y