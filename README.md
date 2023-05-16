# GPU-Cuda Image Processing Tools

This project implements a batch image processing tool using CUDA C. The tool reads a list of image processing jobs from an input text file, applies the specified image processing algorithm to each image using CUDA kernels, and saves the processed images to disk. The project includes implementations of six image processing algorithms: brightness adjustment, grayscale conversion, channel correction, blurring, sharpening, and edge detection.

## Requirements

- NVIDIA CUDA Toolkit 10.0 or later
- libpng

## Running

The project can be run using the following command:

> ./runner.sh

All the output images will be found in `/output`

## Implementation Details

- CUDA threads are allocated in a 2D grid of 2D blocks, where blocks are square and with parametrized size.
- Memory tiling optimization is used to optimize memory management overhead for GPU memory allocation and copying.
- Image format of the pictures is supposed in PNG on 3 Bytes, 1 Byte per RGB channel.
- Processing algorithms are implemented using CUDA kernels.
- The six implemented algorithms are:

  - Brightness adjustment
  - Grayscale conversion
  - Channel correction
  - Blurring
  - Sharpening
  - Edge detection
