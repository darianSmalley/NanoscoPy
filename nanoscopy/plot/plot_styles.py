import os

# Get full directory path for the plot folder.
dir_path = os.path.dirname(os.path.realpath(__file__))

style_dict = {'jetplot': os.path.join(dir_path, 'matplotlib_styles\jetplot.mplstyle') , 'jet-paper': os.path.join(dir_path, 'matplotlib_styles\jet-paper.mplstyle')}
