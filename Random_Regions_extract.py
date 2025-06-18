import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading

# Disable PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

class TiffSectionExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Section Extractor")
        self.root.geometry("600x500")  # Increased window size
        
        # Variables
        self.input_files = []  # List to store multiple selected files
        self.output_path = tk.StringVar(value="extracted_sections")
        self.section_size = tk.IntVar(value=512)
        self.num_sections = tk.IntVar(value=6)
        self.use_random = tk.BooleanVar(value=True)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input selection
        ttk.Label(main_frame, text="Input Files:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(input_frame, text="Select Multiple Files", 
                  command=self.select_multiple_files).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(input_frame, text="Select Directory", 
                  command=self.select_directory).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(input_frame, text="Clear Selection", 
                  command=self.clear_selection).grid(row=0, column=2)
        
        # Selected files status (no detailed list to avoid overflow)
        self.file_status = tk.StringVar(value="No files selected")
        ttk.Label(main_frame, textvariable=self.file_status, 
                 foreground="blue").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_path, 
                 width=50).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="Browse", 
                  command=self.select_output_dir).grid(row=0, column=1, padx=(10, 0))
        
        output_frame.columnconfigure(0, weight=1)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Extraction Settings", padding="15")
        settings_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 10))
        
        # Section size
        ttk.Label(settings_frame, text="Section Size (pixels):").grid(row=0, column=0, sticky=tk.W, pady=5)
        size_spin = ttk.Spinbox(settings_frame, from_=64, to=2048, increment=64, 
                               textvariable=self.section_size, width=12)
        size_spin.grid(row=0, column=1, sticky=tk.W, padx=(15, 0), pady=5)
        
        # Number of sections
        ttk.Label(settings_frame, text="Number of Sections:").grid(row=1, column=0, sticky=tk.W, pady=5)
        num_spin = ttk.Spinbox(settings_frame, from_=1, to=20, 
                              textvariable=self.num_sections, width=12)
        num_spin.grid(row=1, column=1, sticky=tk.W, padx=(15, 0), pady=5)
        
        # Random regions checkbox
        ttk.Checkbutton(settings_frame, text="Use random regions (recommended)", 
                       variable=self.use_random).grid(row=2, column=0, columnspan=2, 
                                                     sticky=tk.W, pady=(15, 5))
        
        # Output format info
        info_label = ttk.Label(settings_frame, 
                              text="Output: Multi-channel TIFF format (ImageJ compatible)", 
                              foreground="gray")
        info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Progress section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 10))
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        progress_frame.columnconfigure(0, weight=1)
        
        # Extract button - now positioned better
        extract_btn = ttk.Button(main_frame, text="Extract TIFF Sections", 
                               command=self.start_extraction, 
                               style="Accent.TButton")
        extract_btn.grid(row=7, column=0, columnspan=2, pady=(20, 10))
        
        # Configure grid weights for proper resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Settings frame can expand
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def select_multiple_files(self):
        filenames = filedialog.askopenfilenames(
            title="Select TIFF Files",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filenames:
            self.input_files = list(filenames)
            self.update_file_status()
            
    def select_directory(self):
        directory = filedialog.askdirectory(title="Select Directory with TIFF Files")
        if directory:
            # Get all TIFF files in the directory
            dir_path = Path(directory)
            tiff_files = list(dir_path.glob("*.tif")) + list(dir_path.glob("*.tiff"))
            self.input_files = [str(f) for f in tiff_files]
            self.update_file_status()
            
    def clear_selection(self):
        self.input_files = []
        self.update_file_status()
        
    def update_file_status(self):
        if not self.input_files:
            self.file_status.set("No files selected")
        elif len(self.input_files) == 1:
            filename = Path(self.input_files[0]).name
            self.file_status.set(f"1 file selected: {filename}")
        else:
            self.file_status.set(f"{len(self.input_files)} files selected")
            
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_path.set(directory)
            
    def start_extraction(self):
        if not self.input_files:
            messagebox.showerror("Error", "Please select input files or directory")
            return
            
        # Run extraction in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.extract_sections)
        thread.daemon = True
        thread.start()
        
    def extract_sections(self):
        self.progress_var.set("Processing...")
        self.progress_bar.start()
        
        try:
            success_count = self.extract_representative_sections(
                input_files=self.input_files,
                output_dir=self.output_path.get(),
                section_size=self.section_size.get(),
                num_sections=self.num_sections.get(),
                use_random=self.use_random.get()
            )
            
            if success_count > 0:
                self.progress_var.set(f"Extraction completed! Processed {success_count} files.")
                messagebox.showinfo("Success", 
                                  f"Sections extracted from {success_count} files to: {self.output_path.get()}")
            else:
                self.progress_var.set("No files processed")
                messagebox.showwarning("Warning", "No files were processed successfully")
                
        except Exception as e:
            self.progress_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.progress_bar.stop()
            
    def extract_representative_sections(self, input_files, output_dir, section_size, 
                                      num_sections, use_random=True):
        """Extract representative sections from TIFF files preserving multichannel structure"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        processed_count = 0
        
        for i, tiff_file_path in enumerate(input_files):
            tiff_file = Path(tiff_file_path)
            self.progress_var.set(f"Processing {i+1}/{len(input_files)}: {tiff_file.name}")
            
            try:
                with Image.open(tiff_file) as img:
                    width, height = img.size
                    
                    # Skip if image is smaller than section size
                    if width < section_size or height < section_size:
                        continue
                    
                    # Check if image has multiple frames/channels
                    is_multiframe = hasattr(img, 'n_frames') and img.n_frames > 1
                    
                    positions = []
                    
                    if use_random:
                        # Generate random positions
                        np.random.seed(42 + i)  # Different seed per file for variety
                        for j in range(num_sections):
                            x = np.random.randint(0, width - section_size)
                            y = np.random.randint(0, height - section_size)
                            positions.append((x, y))
                    else:
                        # Use corners and center (old behavior)
                        positions.append((0, 0))  # Top-left
                        if num_sections > 1:
                            positions.append((width - section_size, 0))  # Top-right
                        if num_sections > 2:
                            positions.append((0, height - section_size))  # Bottom-left
                        if num_sections > 3:
                            positions.append((width - section_size, height - section_size))  # Bottom-right
                        if num_sections > 4:
                            center_x = (width - section_size) // 2
                            center_y = (height - section_size) // 2
                            positions.append((center_x, center_y))
                    
                    # Extract and save sections preserving multichannel structure
                    base_name = tiff_file.stem
                    for j, (x, y) in enumerate(positions[:num_sections]):
                        
                        if is_multiframe:
                            # Handle multichannel TIFF
                            sections = []
                            n_frames = img.n_frames
                            
                            for frame_idx in range(n_frames):
                                img.seek(frame_idx)
                                frame = img.copy()
                                section = frame.crop((x, y, x + section_size, y + section_size))
                                sections.append(section)
                            
                            # Save multichannel TIFF
                            section_name = f"{base_name}_section_{j+1:02d}_x{x}_y{y}.tif"
                            section_path = output_path / section_name
                            
                            # Save first frame
                            sections[0].save(
                                section_path, 
                                format='TIFF',
                                save_all=True,
                                append_images=sections[1:] if len(sections) > 1 else [],
                                compression='lzw'  # Use LZW compression for smaller files
                            )
                            
                        else:
                            # Handle single channel TIFF
                            section = img.crop((x, y, x + section_size, y + section_size))
                            section_name = f"{base_name}_section_{j+1:02d}_x{x}_y{y}.tif"
                            section_path = output_path / section_name
                            section.save(section_path, format='TIFF', compression='lzw')
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {tiff_file.name}: {e}")
                continue
        
        return processed_count

def main():
    root = tk.Tk()
    
    # Try to use modern styling if available
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except:
        pass
    
    app = TiffSectionExtractor(root)
    root.mainloop()

if __name__ == "__main__":
    main()