import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Function to open file dialog and select images
def select_images():
    file_paths = filedialog.askopenfilenames(
        title="Select up to 4 images",
        filetypes=[("Image files", "*.tiff', *.tf2")],
        multiple=True
    )
    # Limit to 4 images
    file_paths = file_paths[:4]
    display_images(file_paths)

# Function to display selected images
def display_images(file_paths):
    for i, file_path in enumerate(file_paths):
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        original_labels[i].config(image=img)
        original_labels[i].image = img
        # Display histogram
        display_histogram(file_path, i)
        # Process and display image
        # process_and_display_image(file_path, i)

# Function to display histogram
def display_histogram(file_path, index):
    img = Image.open(file_path).convert('L')
    hist = img.histogram()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(hist)
    canvas = FigureCanvasTkAgg(fig, master=histogram_frames[index])
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to process and display image
def process_and_display_image(file_path, index):
    img = Image.open(file_path)
    # Example processing: convert to grayscale
    processed_img = img.convert('RGB')
    processed_img = processed_img.resize((150, 150))
    processed_img = ImageTk.PhotoImage(processed_img)
    processed_labels[index].config(image=processed_img)
    processed_labels[index].image = processed_img

# Initialize the main window
root = tk.Tk()
root.title("Image Processor")

# Add a button to select images
select_button = tk.Button(root, text="Select Images", command=select_images)
select_button.pack(pady=20)

# Add frames and labels to display original images, histograms, and processed images
original_labels = [tk.Label(root) for _ in range(4)]
histogram_frames = [tk.Frame(root) for _ in range(4)]
processed_labels = [tk.Label(root) for _ in range(4)]

for i in range(4):
    original_labels[i].pack(side=tk.LEFT, padx=10)
    histogram_frames[i].pack(side=tk.LEFT, padx=10)
    processed_labels[i].pack(side=tk.LEFT, padx=10)

# Add widgets to change processing parameters (example: threshold slider)
threshold_label = tk.Label(root, text="Threshold:")
threshold_label.pack()
threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL)
threshold_slider.pack()

# Start the Tkinter main loop
root.mainloop()