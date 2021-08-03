import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas as pd
import numpy as np
import os

class PlotGen():
    def __init__(self , data , data_labels = None , axis_labels = None , scale_factors = None):
        """
        data: Numpy array of 2-column Numpy containing one sub-array representing each curve to be plotted. The first column of each subarray should be the X axis data and the second the Y axis data.
        data_labels: Numpy array of strings. Used to specify legend entries for plotted curves.
        axis_labels: Dictionary. Should contain key:value pairs including either or both of the following: 'xlabel':'X Axis Title' , 'ylabel':'Y Axis Title'
        scale_factors: List. Should contain two numbers which represent the factors by which the X and Y values of data should be scaled by before plotting. This is useful for converting between units on plots.
        """
        # Define inputs as class properties for later use
        self.data = data
        self.data_labels = data_labels
        self.axis_labels = axis_labels
        # Initialize matplotlib figure and axes objects
        self.fig , self.ax = plt.subplots()
        # Construct a plot from the data inputs using the 'default' style
        self.set_scalefactors(scale_factors)
        self.plot_default()
    
    def set_scalefactors(self, scale_factors):
        """
        Determine whether there should be a scale factor for us in plotting.
        """
        if isinstance(scale_factors , list):
            if isinstance(scale_factors[0] , (int , float)):
                self.scale_factors = scale_factors[:2]
        else:
            self.scale_factors = [1,1]
    
    def plot_default(self):
        """
        Plot the data using the Ishigami group 'default' style
        """
        #### Add data to the axes ####
        # Make the base plot the data
        # Case for a single set of data
        if isinstance(self.data[0] , np.ndarray):
            self.ax.plot(self.data[0] * self.scale_factors[0] , self.data[1] * self.scale_factors[1] , 'o-')
        # Case for multiple sets of data
        elif isinstance(self.data[0] , list):
            for curve_data in self.data:
                self.ax.plot(curve_data[0] * self.scale_factors[0] , curve_data[1] * self.scale_factors[1] , 'o-')
        
        # Apply specified axis labels (if applicable)
        if self.axis_labels != None:
            self.ax.set(**self.axis_labels)
        
        # Create/update the legend (if applicable)
        self.update_legend()
        
        #### Adjust styles of the plot to the Ishigami Group 'default' ####
        self.fig.set_size_inches(8,6)
        self.ax.grid(linestyle = '--' , alpha=0.5)
        
        # Adjust ticks and tick labels
        self.ax.xaxis.set_minor_locator(AutoMinorLocator()) # Add minor ticks to x axis
        self.ax.yaxis.set_minor_locator(AutoMinorLocator()) # Add minor ticks to y axis
        self.ax.tick_params(which = 'both',bottom = True , top = True , left = True , right = True) # Add ticks to top and right spines
        self.ax.tick_params(which='both', direction='in', labelsize=16) # Adjust tick appearance
        self.ax.xaxis.label.set_size(20)
        self.ax.yaxis.label.set_size(20)
        
    def update_legend(self):
        # Create/update the legend (if applicable)
        if self.data_labels != None:
            if isinstance(self.data_labels , str):
                self.ax.get_lines()[0].set_label(self.data_labels)
            elif isinstance(self.data_labels , list):
                for index , line in enumerate(self.ax.get_lines()):
                    line.set_label(self.data_labels[index])
            self.ax.legend(fontsize = 16)
            
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
        self.fig.set_tight_layout(True)
        self.fig.savefig( output_path , facecolor = 'white', transparent=False)

#######################################################
########### Notes for improving PlotGen ###############
# Look into the Matplotlib Styles functionality and make sure that this class is still sensible.
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