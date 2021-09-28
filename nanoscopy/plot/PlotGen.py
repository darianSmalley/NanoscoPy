import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas as pd
import numpy as np
import os

class PlotGen():
    def __init__(self , data , data_labels = None , axis_labels = None , scale_factors = None , plot_styles = 'default'):
        """
        data: Numpy array or list of 2-column Numpy arrays containing one sub-array representing each curve to be plotted. The first column of each subarray should be the X axis data and the second the Y axis data.
        data_labels: String or list of strings. Used to specify legend entries for plotted curves.
        axis_labels: Dictionary. Should contain key:value pairs including either or both of the following: 'xlabel':'X Axis Title' , 'ylabel':'Y Axis Title'
        scale_factors: List. Should contain two numbers which represent the factors by which the X and Y values of data should be scaled by before plotting. This is useful for converting between units on plots.
        plot_styles: string or list of strings. The strings should be either names of stylesheets present in the Matplotlib directory or a full path to a stylesheet in another location.
        """
        # Define inputs as class properties for later use
        self.data = data
        self.data_labels = data_labels
        self.axis_labels = axis_labels
        self.plot_styles = plot_styles
        
        # Initialize matplotlib figure and axes objects using specified matplotlib stylesheet(s)
        with plt.style.context(self.plot_styles):
            self.fig , self.ax = plt.subplots()
        
        # Construct a plot from the data inputs using the 'default' style
        self.set_scalefactors(scale_factors)
        self.plot_default()
    
    def set_scalefactors(self, scale_factors):
        """
        Sets scale factors by which data for plotting is multiplied. This might be useful when plotting very large or very small quantities.
        
        Input:
            scale_factors: list of floats (should be of length 2) or None.
                If scale_factors is None: no scaling occurs.
                If scale_factors is a 2-element list of floats: X axis data is multiplied by the first element and the Y data is multiplied by the second.
        """
        # Check to see if a list of scale factors is provided.
        if isinstance(scale_factors , list):
            # Make sure that the elements are suitable for multiplication
            if isinstance(scale_factors[0] , (int , float)):
                # Store the scale factors as an attribute.
                self.scale_factors = scale_factors[:2]
        else:
            # If the provided scale factors are unsuitable, set them to 1 and thus not scale the data.
            self.scale_factors = [1,1]
    
    def plot_default(self):
        """
        Plot the data using the Ishigami group 'default' style
        """
        # Use the specified
        with plt.style.context(self.plot_styles):
            #### Add data to the axes ####
            # Make the base plot the data
            # Case for a single set of data
            if isinstance(self.data[0] , np.ndarray):
                self.ax.plot(self.data[0] * self.scale_factors[0] , self.data[1] * self.scale_factors[1])
            # Case for multiple sets of data
            elif isinstance(self.data , list):
                for curve_data in self.data:
                    self.ax.plot(curve_data[0] * self.scale_factors[0] , curve_data[1] * self.scale_factors[1])
            
            # Apply specified axis labels (if applicable)
            if self.axis_labels != None:
                self.ax.set(**self.axis_labels)
            
            # Apply a legend (if applicable)
            self.update_legend()
            
            # Adjust ticks and tick labels
            self.ax.xaxis.set_minor_locator(AutoMinorLocator()) # Add minor ticks to x axis
            self.ax.yaxis.set_minor_locator(AutoMinorLocator()) # Add minor ticks to y axis
    
    ##### Convenience Functions ######
    def update_legend(self):
        # Create/update the legend (if applicable)
        if self.data_labels != None:
            # The case for only one curve.
            if isinstance(self.data_labels , str):
                # Get the curve object from the plot and assign the label to it
                self.ax.get_lines()[0].set_label(self.data_labels)
            elif isinstance(self.data_labels , list):
                # The case for multiple curves on a single plot.
                # Get a list of curves and assign labels to them one at a time.
                for index , line in enumerate(self.ax.get_lines()):
                    line.set_label(self.data_labels[index])
            # Update the legend with the new data labels
            self.ax.legend()
            
    def update_linestyles(self, linestyles):
        """
        linestyles: Either a string or a list of strings.
        
        Adjusts the line styles of each data set shown on a plot. If linestyles is a string, the specified line style will be applied to all data sets in the plot. If linestyles is a list it should have the same length as the number of data sets on the plot and each line will be given the style specified with the same index in linestyles.
        """
        # Case: Either only 1 set of data being plotted or multiple sets to be plotted with the same linestyle
        if isinstance(linestyles , str):
            for line in self.ax.get_lines():
                line.set_linestyle(linestyles)
        # Case: Multiple sets of data being plotted, specifying linestyle for each set individually
        elif isinstance(linestyles , list):
            for index , line in enumerate(self.ax.get_lines()):
                line.set_linestyle(linestyles[index])
        self.update_legend()
    
    def update_markerstyles(self, markerstyles):
        """
        markerstyles: Either a string or a list of strings.
        
        Adjusts the marker styles of each data set shown on a plot. If markerstyles is a string, the specified marker style will be applied to all data sets in the plot. If markerstyles is a list it should have the same length as the number of data sets on the plot and each line's markers will be given the style specified with the same index in markerstyles.
        """
        # Case: Either only 1 set of data being plotted or multiple sets to be plotted with the same linestyle
        if isinstance(markerstyles , str):
            for line in self.ax.get_lines():
                line.set_marker(markerstyles)
        # Case: Multiple sets of data being plotted, specifying linestyle for each set individually
        elif isinstance(markerstyles , list):
            for index , line in enumerate(self.ax.get_lines()):
                line.set_marker(markerstyles[index])
        self.update_legend()
    
    def export_fig(self , output_path):
        """
        output_path: string. Specifies the path to which the figure will be exported. The path must include a filename and valid image extension.

        Saves the figure in the PlotGen object to a file. 
        """
        self.fig.set_tight_layout(True)
        self.fig.savefig( output_path , facecolor = 'white', transparent=False)

#######################################################
########### Notes for improving PlotGen ###############
# Add the ability to supply PlotGen with the fig , ax of an existing plot and then have it update the style of the existing plot in place.
# Encapsulate more of the plot_default actions into separate methods to facilitate plot updates in addition to plot creation.

# Tutorial Thoughts: Make sure to emphasize that PlotGen is really just a wrapper around the object-oriented interface of Matplotlib. This means that anything that can be done on a normal plot's ax can also be done on PlotGen.ax.
# Tutorial Thoughts: The above comment means that a user doesn't need to make all style changes to a plot at once or even at the call to PlotGen.

#######################################################

class TabularData():
    def __init__(self , filepath , data_names = None , x_baseline = None , y_baseline = None , preview_plot = False):
        """
        filepath: string or list.
        data_names: list.
        """
        self.x_baseline = x_baseline
        self.y_baseline = y_baseline
        self.data = self.load_data(filepath)
        self.data_names = data_names
        self.set_data()
        if self.data_names != None:
            self.subtract_baseline()
        self.plot_data = [self.xdata , self.ydata]
        if preview_plot == True:
            self.preview_plot = PlotGen( [self.plot_data] , axis_labels = {'xlabel':self.data_names[0] , 'ylabel':self.data_names[1]})
    
    def load_data(self,filepath , sep = '\t'):
        """
        Load tabular data
        filepath: String or list. The filepath of the file to be loaded can be specified as a complete path as a string or a list of snipets of the string to be concatonated.
        sep: string. Specifies the delimeter used in the file.
        """
        if isinstance(filepath, list): # Determine if the path was specified as a list of path snipets
            filepath = os.path.join(*filepath) # Use list unpacking to separate elements into multiple arguments
        return pd.read_csv(filepath , sep)
    
    def set_data(self):
        """
        Define specific as the X and Y data for the class object (for convenience).
        """
        if self.data_names != None:
            self.xdata = self.data[self.data_names[0]].to_numpy()
            self.ydata = self.data[self.data_names[1]].to_numpy()
    
    def subtract_baseline(self):
        if self.x_baseline == 'min':
            self.xdata = self.xdata - np.min(self.xdata)
        elif isinstance(self.x_baseline, (int, float)):
            self.xdata = self.xdata - self.x_baseline
        
        if self.y_baseline == 'min':
            self.ydata = self.ydata - np.min(self.ydata)
        elif isinstance(self.y_baseline, (int, float)):
            self.ydata = self.ydata - self.y_baseline

#######################################################
######### Notes for improving TabularData #############


#######################################################
#################### Other Ideas ######################
# It would be nice to have a way to handle datasets. This could be a new class which is essentially a list or disctionary of TabularData objects. The user could input a list of filenames or paths and labels for the data
# to load and the class could handle the loading. Then the user could issue a single command to have it make a summary plot of all the data in the set.
# Put the TabularData class in the dataIO package.
# Need to either remove the call to PlotGen in the TabularData class or to import PlotGen from its parent package.