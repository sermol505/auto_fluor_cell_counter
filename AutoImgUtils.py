import numpy as np
from skimage import measure, draw, color, feature
from skimage.segmentation import watershed, find_boundaries
from skimage.filters import threshold_otsu, threshold_local, threshold_mean, threshold_minimum
import tkinter as tk
from tkinter.filedialog import askopenfilename,askdirectory
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import math

def select_file():
    """
    This function opens a dialog box to select a file path.

    Returns:
    str: The file path selected
    """
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = askopenfilename(title="Select Reference Image File")
    root.destroy()
    return file_path

def select_files():
    """
    This function opens a dialog box to select multiple file paths.
    
    Returns:
    list: A list of file paths selected by the user.
    """
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_list = []
    
    while True:
        file = askopenfilename(title="Select Reference Image File")
        if file:
            file_list.append(file)
        else:
            break  # User cancelled the dialog
    
    root.destroy()
    return list(set(file_list))  # Remove duplicates

def select_folder():
    """
    This function opens a dialog box to select a folder path.

    Returns:
    str: The folder path selected
    """
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    folder_path = askdirectory(title="Select Folder with Channel TIFF Images")
    root.destroy()
    return folder_path

def select_folders():
    """
    This function opens a dialog box to select multiple folder paths.
    
    Returns:
    list: A list of folder paths selected by the user.
    """
    root = tk.Tk()
    root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
    folder_list = []
    
    while True:
        folder = askdirectory(title="Select Folder Containing Channel TIFF Images")
        if folder:
            folder_list.append(folder)
        else:
            break  # User cancelled the dialog
    
    root.destroy()
    return list(set(folder_list))  # Remove duplicates

def normalize_img(image, method = 'minmax'):
    """
    This function normalizes a given image array individually per channel.
    
    Parameters:
    image (numpy.ndarray): A numpy array representing the image to be normalized.
    method (str): The normalization method to use. Default is 'minmax'.
        'minmax' scales the pixel values to be between 0 and 1.
        'zscore' standardizes the pixel values to have a mean of 0 and a standard deviation of 1.
        'percentile' scales the pixel values to be between the 1st and 99th percentiles.

    Returns:
    numpy.ndarray: The normalized image array with pixel values between 0 and 1.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension if not present
        sing_ch = True
    else:
        sing_ch = False

    for ch in range(image.shape[-1]):
        image[..., ch] = image[..., ch].astype(np.float32)  # Ensure float type for division

        if method == 'minmax':
            min_val = np.min(image[..., ch])
            max_val = np.max(image[..., ch])
            if max_val - min_val > 0:
                image[..., ch] = (image[..., ch] - min_val) / (max_val - min_val)
            else:
                image[..., ch] = 0
        elif method == 'zscore':
            mean_val = np.mean(image[..., ch])
            std_val = np.std(image[..., ch])
            if std_val > 0:
                image[..., ch] = (image[..., ch] - mean_val) / std_val
            else:
                image[..., ch] = 0
        elif method == 'percentile':
            p1 = np.percentile(image[..., ch], 1)
            p99 = np.percentile(image[..., ch], 99)
            if p99 - p1 > 0:
                image[..., ch] = (image[..., ch] - p1) / (p99 - p1 + 1e-8) # Avoid division by zero
            else:
                image[..., ch] = 0
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
    if sing_ch:
        image = np.squeeze(image, axis=-1)  # Remove channel dimension if single channel image

    return image

def bg_substraction_ROI(image, background_threshold=None, display_rois=True):
    """
    Perform background subtraction on a multi-channel image using region of interest (ROI) analysis.
    Parameters:
    -----------
    image : numpy.ndarray
        The input image with multiple channels (e.g., RGB or multi-spectral image).
    background_threshold : float, optional
        The threshold value to define background regions. If None, it will be calculated based on the mean and standard deviation of each channel.
    display_rois : bool, optional
        If True, display the background ROIs on each channel.
    Returns:
    --------
    background_values : dict
        A dictionary containing the mean background values for each channel.
    mean_background_value : list
        A list of mean background values for each channel.
    background_subtracted_image : numpy.ndarray
        The background-subtracted image.
    
    """
    
    n_channels = image.shape[2]

    # Initialize empty list to store background values
    background_values = {f'Channel {i+1}': [] for i in range(n_channels)}
    
    # Initialize empty image to store background subtracted image
    background_subtracted_image = np.copy(image)

    background_threshold_temp = background_threshold
    
    for channel_index in range(n_channels):
        channel = image[:, :, channel_index]
        # Define background threshold, by standard deviation from the mean, if not provided
        if background_threshold_temp == None:    
            background_threshold_temp = np.mean(channel) - 0.2 * np.std(channel)
            if background_threshold_temp < 0:
                background_threshold_temp = np.min(channel) + 0.01 * np.std(channel)

        background_thresh = channel < background_threshold_temp
        print(f'Background threshold for channel {channel_index+1}: {background_threshold}')

        # Label the background regions
        background_labels = measure.label(background_thresh)
        background_props = measure.regionprops(background_labels)

        # Create bounding boxes around the detected background contours to define background ROIs
        background_rois = []

        for prop in background_props:
            minr, minc, maxr, maxc = prop.bbox[:4]
            background_rois.append((minc, minr, maxc - minc, maxr - minr))

        # Calculate mean background values for the channel
        for roi in background_rois:
            x, y, w, h = roi
            roi_area = channel[y:y+h, x:x+w]
            mean_background = np.mean(roi_area)
            background_values[f'Channel {channel_index+1}'].append(mean_background)
        
        if display_rois:
            # Display background ROIs on the current channel
            roi_image = np.stack([normalize_img(channel)]*3, axis=-1)  # Convert to RGB
            for roi in background_rois:
                x, y, w, h = roi
                rr, cc = draw.rectangle_perimeter((y, x), extent=(h, w), shape=channel.shape)
                roi_image[rr, cc] = [1, 0, 0]  # Red for ROIs

            plt.imshow(roi_image)
            plt.title(f'Background ROIs on Channel {channel_index + 1}')
            plt.show()
    
    # Subtract background from the entire image
    mean_background_value = []
    for channel_index in range(n_channels):
        mean_background_value.append(np.mean(background_values[f'Channel {channel_index+1}']))
        print(f'Mean background value for channel {channel_index+1}: {mean_background_value[channel_index]}')
        background_subtracted_image[:, :, channel_index] -= np.minimum(math.floor(mean_background_value[channel_index]), background_subtracted_image[:, :, channel_index])

    return background_values, mean_background_value, background_subtracted_image

def bg_substraction_ROI_single_ch(image, background_threshold = None, channel_of_interest = 0, display_rois = False):
    
    """
    Perform background subtraction on a multi-channel image using region of interest (ROI) based on one of the channel_of_interest.
    Utilizes the chosen channel as a mask to find the darkests spots in the image according to background_threshold. 
    Parameters:
    -----------
    image : numpy.ndarray
        The input image with multiple channels (e.g., RGB or multi-spectral image).
    background_threshold : float, optional
        The threshold value to define background regions. If None, it will be calculated based on the mean and standard deviation of each channel.
    display_rois : bool, optional
        If True, display the background ROIs on each channel.
    Returns:
    --------
    background_values : dict
        A dictionary containing the mean background values for each channel.
    mean_background_value : list
        A list of mean background values for each channel.
    background_subtracted_image : numpy.ndarray
        The background-subtracted image.
    
    """

    n_channels = image.shape[2]

    # Initialize empty list to store background values
    background_values = {f'Channel {i+1}': [] for i in range(n_channels)}
    
    # Initialize empty image to store background subtracted image
    bg_subs_image_single_ch = image[:,:,channel_of_interest]

    # Define background threshold, by standard deviation from the mean, if not provided
    if background_threshold is None:
            background_threshold = np.mean(bg_subs_image_single_ch) - 0.2 * np.std(bg_subs_image_single_ch)
            # print(f'Background threshold for channel {channel_of_interest+1}: {background_threshold}')
            if background_threshold < 0:
                background_threshold = np.min(bg_subs_image_single_ch) + 0.01 * np.std(bg_subs_image_single_ch)
            print(f'Background threshold for channel {channel_of_interest+1}: {background_threshold}')
    
    # Find minima below a threshold to define background ROIs

    background_thresh = bg_subs_image_single_ch < background_threshold

    # Label the background regions
    background_labels = measure.label(background_thresh)
    background_props = measure.regionprops(background_labels)

    # Create bounding boxes around the detected background contours to define background ROIs
    background_rois = []

    for prop in background_props:
        minr, minc, maxr, maxc = prop.bbox[:4]
        background_rois.append((minc, minr, maxc - minc, maxr - minr))
    
    if display_rois: 
        # Display background ROIs on channel of interest
        background_roi_image = np.stack([normalize_img(bg_subs_image_single_ch)]*3, axis=-1)  # Convert to RGB

        for roi in background_rois:
            x, y, w, h = roi
            background_roi_image[y:y+h, x:x+w] = [1, 0, 0]  # Red for background ROIs

        plt.imshow(background_roi_image)
        plt.title(f'Background ROIs on Channel of Interest {channel_of_interest + 1}')
        plt.show()

    # Calculate mean background values for each channel
    for channel_index in range(n_channels):
        channel = image[:, :, channel_index]
        
        for roi in background_rois:
            x, y, w, h = roi
            roi_area = channel[y:y+h, x:x+w]
            mean_background = np.mean(roi_area)
            background_values[f'Channel {channel_index+1}'].append(mean_background)

    mean_background_value = []
    # Subtract background from the entire image
    background_subtracted_image = np.copy(image)
    
    for channel_index in range(n_channels):
        mean_background_value.append(np.mean(background_values[f'Channel {channel_index+1}']))
        background_subtracted_image[:, :, channel_index] -= np.nan_to_num(np.minimum(math.floor(mean_background_value[channel_index]), background_subtracted_image[:, :, channel_index]))
    
    return background_values, mean_background_value, background_subtracted_image

def watershed_rois(binary):
    """
    Enhanced watershed segmentation for neuron detection.
    
    Parameters:
    -----------
    binary : numpy.ndarray
        Binary image after initial thresholding
    
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image where each neuron has a unique integer value
    """
    # Distance transform to separate touching cells
    distance = ndi.distance_transform_edt(binary)
    
    # Find local maxima (cell centers)
    local_max = feature.peak_local_max(distance, labels=binary, footprint=np.ones((3, 3)))
    
    # Create markers for watershed
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_max.T)] = True
    markers, _ = ndi.label(mask)

    # Apply watershed
    labels = watershed(
        -distance,  # Negative distance to find boundaries
        markers, 
        mask=binary,
        watershed_line=True  # Include separation lines
    )
    
    return labels

def draw_ellipse_roi(props):
    y, x = props.centroid
    orientation = props.orientation
    major_axis = props.major_axis_length / 2
    minor_axis = props.minor_axis_length / 2
    rr, cc = draw.ellipse(y, x, minor_axis, major_axis, rotation=orientation)
    return rr, cc

def get_contour_rois(binary_image):
    contours = measure.find_contours(binary_image, 0.5)
    rois = []
    for contour in contours:
        rois.append(contour)
    return rois

def process_rois(binary_image, roi_type='watershed'):
    """
    Process binary image and return consistent ROI information regardless of detection method.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary image after thresholding
    roi_type : str
        Type of ROI detection to use
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image where each ROI has unique integer value
    props : list
        List of region properties from skimage.measure.regionprops
    rois : list
        List of ROI coordinates/boundaries
    """
    if roi_type == 'watershed':
        labels = watershed_rois(binary_image)
        props = measure.regionprops(labels)
        rois = [(prop.bbox[1], prop.bbox[0], 
                prop.bbox[3] - prop.bbox[1], 
                prop.bbox[2] - prop.bbox[0]) for prop in props]
    
    elif roi_type == 'contour':
        contours = measure.find_contours(binary_image, 0.5)
        # Create label image from contours
        labels = np.zeros_like(binary_image, dtype=int)
        for i, contour in enumerate(contours, start=1):
            rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
            labels[rr, cc] = i
        props = measure.regionprops(labels)
        rois = contours
    
    elif roi_type == 'ellipse':
        labels = measure.label(binary_image)
        props = measure.regionprops(labels)
        rois = []
        for prop in props:
            rr, cc = draw_ellipse_roi(prop)
            # Only keep valid coordinates
            mask = (rr >= 0) & (rr < binary_image.shape[0]) & \
                  (cc >= 0) & (cc < binary_image.shape[1])
            rois.append(np.column_stack((rr[mask], cc[mask])))
    
    else:
        raise ValueError(f"Unsupported ROI type: {roi_type}")
        
    return labels, props, rois

def plot_rois(image, labels, props, rois, roi_type, title=None):
    """
    Display ROIs with consistent visualization across detection methods.
    """
    display_img = np.stack([normalize_img(image)]*3, axis=-1)
    
    # Draw boundaries based on labels
    boundaries = find_boundaries(labels)
    display_img[boundaries] = [1, 0, 0]  # Red boundaries
    
    # Add ROI numbers
    for i, prop in enumerate(props, start=1):
        y, x = map(int, prop.centroid)
        plt.text(x, y, str(i), color='white', fontsize=8)
    
    plt.imshow(display_img)
    plt.title(title or f'ROIs using {roi_type} detection')
    plt.show()

def thresholding(image, th_method='otsu', roi_type='watershed', display_rois=False):
    """
    Perform thresholding on an image using the specified method and return the segmented image and its ROIs.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be thresholded.
    method : str, optional
        The thresholding method to use. Options are 'otsu', 'adaptive', 'mean', 'minimum', etc.
    roi_type : str, optional
        The type of ROI to use. Options: 'rectangle', 'ellipse', 'contour', 'convex_hull', 'watershed'
    display_rois : bool, optional
        If True, display the ROIs on the segmented image.
    
   Returns:
    --------
    binary : numpy.ndarray
        The binary segmented image
    labels : numpy.ndarray
        Labeled image where each ROI has unique integer value
    props : list
        List of region properties
    rois : list
        List of ROI coordinates/boundaries
    """

    # Convert image to grayscale
    gray_image = normalize_img(np.mean(image, axis=2))
    
    # Thresholding
    if th_method == 'otsu':
        thresh = threshold_otsu(gray_image)
        binary = gray_image > thresh
    elif th_method == 'adaptive':
        block_size = 10
        binary = gray_image > threshold_local(gray_image, block_size, offset=10)
    elif th_method == 'mean':
        thresh = threshold_mean(gray_image)
        binary = gray_image > thresh
    elif th_method == 'minimum':
        thresh = threshold_minimum(gray_image)
        binary = gray_image > thresh
    else:
        raise ValueError(f"Unsupported thresholding method: {th_method}")

    # Display original, grayscale, and binary images side by side
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(normalize_img(image[:, :, :3]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.subplot(133)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.show()

    # Clean up binary image
    binary = ndi.binary_fill_holes(binary)
    # binary = ndi.binary_opening(binary, structure=np.ones((3,3)))

    # Process ROIs with consistent output
    labels, props, rois = process_rois(binary, roi_type)
    
    if display_rois:
        plot_rois(gray_image, labels, props, rois, roi_type)

    return binary, labels, props, rois
