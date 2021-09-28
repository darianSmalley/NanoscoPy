import os

# Get full directory path for the plot folder.
dir_path = os.path.dirname(os.path.realpath(__file__))

# Generate a dictionary containing the paths to custom Matplotlib stylesheets. This is used for importing the stylesheets into Matplotlib
style_dict = {'jetplot': os.path.join(dir_path, 'matplotlib_styles\jetplot.mplstyle')}