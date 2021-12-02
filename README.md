# NanoscoPy
An open source repo for the analysis of experimental microscopy data common in materials and surface science.

## Example Usage
```
    import nanoscopy as nano
    import matplotlib.pyplot as plt

    # Specify the file to be imported
    filepath_sxm = '../ExampleDataFiles/STM_Au-111_Flat.sxm'

    # Read the data
    scan = nano.spm.io.read(filepath_sxm)[0]

    # Select the forward pass of the height channel from the data
    image = scan.data['Z'][0]

    # Correct the data
    flattened_image = nano.spm.process.basic_flatten(image)

    # Show flattened image 
    plt.imshow(flattened_image)
```