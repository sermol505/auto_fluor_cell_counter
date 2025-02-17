## Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from tkinter import Tk
import tkinter
import imagecodecs
from tkinter.filedialog import askdirectory
import os
from skimage import filters, measure, morphology, draw
import math
import pandas as pd

# Use tkinter to open a file dialog and select the folder with images
def select_folder():
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    folder_path = askdirectory(title="Select Folder with Channel TIFF Images")
    return folder_path

folder_path = select_folder()

# Normalize images for display
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def get_images(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    # Get only tiff files
    files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff')or f.endswith('.tf2')]
    # Sort files by channel
    files.sort()
    # Initialize empty list to store images
    images = []
    # Loop through all files
    for file in files:
        # Read image
        image = tiff.imread(folder_path + '/' + file)
        # Append image to list
        images.append(image)
    return images
