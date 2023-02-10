# NanoscoPy
An open source repo for the analysis of experimental microscopy data common in materials and surface science.

## Example Usage
```python
from nanoscopy import spm
import matplotlib.pyplot as plt

# Specify the SXM or MTRX file to be imported
filepath_sxm = '../ExampleDataFiles/STM_Au-111_Flat.sxm'

# Read the data
scan = spm.read(filepath_sxm)

# Select the forward pass of the height channel from the data
fwd_scan = scan.dataframe.at[0, 'image']

# Correct each image by globally flattened via plane correction, followed by 2nd order polynomial background subtraction, line-by-line offset flattening, 3x3 gaussian smoothing, and contrast limited adaptive histogram equilization.
corrected_scan = spm.correct(fwd_scan, poly=True, equalize=True)

# Show flattened image 
plt.imshow(corrected_scan)
plt.show()
```