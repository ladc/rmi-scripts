#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot data from a RADQPE HDF5 file.')
parser.add_argument('file_path', type=str, help='Path to the HDF5 file')
args = parser.parse_args()

# Open the HDF5 file
with h5py.File(args.file_path, 'r') as f:
    # Navigate to the dataset
    data = f['/dataset1/data1/data'][:]

# Check if the data is 2D (700x700), and plot it
if data.ndim == 2 and data.shape == (700, 700):
    print("Sample Data:", data[345:355, 345:355])  # Print some sample values
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Data Value')
    plt.title('Data from HDF5 File')
    plt.show()
else:
    print(f"Unexpected data shape: {data.shape}")
