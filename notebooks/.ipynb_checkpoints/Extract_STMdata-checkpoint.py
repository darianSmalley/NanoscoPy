root = Tk()
Label(root, text="Select data folder")
folder_path = filedialog.askdirectory()
root.destroy()
mtrx_files = glob.glob(os.path.join(folder_path, "**/*.*Z_mtrx"), recursive = True)

fig, axs = plt.subplots(len(mtrx_files), 2)
fig.suptitle(f'Sample Date: {mtrx_files[0][43:53]}')
fig.set_figheight(15)
fig.set_figwidth(7)

im = spiepy.Im()
mtrx = access2thematrix.MtrxData()
for i, filename in enumerate(mtrx_files):
    traces, message = mtrx.open(filename)
    image, message = mtrx.select_image(traces[0])
    im.data = image.data
    im_output, _ = spiepy.flatten_xy(im)
    image_poly , _ = spiepy.flatten_poly_xy(im_output , deg = 2)

    sample_name = f'{filename[-16:]}'
    axs[i, 0].imshow(im.data , cmap = spiepy.NANOMAP, origin = 'lower')
    axs[i, 0].set_title(sample_name)
    axs[i, 1].imshow(image_poly.data , cmap = spiepy.NANOMAP, origin = 'lower')
    axs[i, 1].set_title('plane & quadratic flattening')
    
#     dst_path = os.path.join(folder_path, filename + ".jpg")
#     plt.imsave(dst_path, img, cmap='gray')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
plt.savefig('WSe2 Post Processing STM Image Comparisions.png', 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none')