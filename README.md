# GPU-Cuda Image Processing Tools

This project implements a batch image processing tool using CUDA C++. The tool reads a list of image processing jobs from an input text file, applies the specified image processing algorithm to each image using CUDA kernels, and saves the processed images to disk. The project includes implementations of six image processing algorithms: brightness adjustment, grayscale conversion, channel correction, blurring, sharpening, and edge detection.

## Requirements

- NVIDIA CUDA Toolkit 10.0 or later
- libpng

## Building

To build the project, run `make` from the project root directory. This will compile the CUDA source code and link the executable. 

## Running

The project can be run using the following command:

> ./image_processing <jobs_file>


where `<jobs_file>` is the path to the input text file containing the list of jobs to be executed. Each line in the file should specify an input image file name, an algorithm name, and an output image file name, separated by spaces.

For example:

> input/image1.png brightness output/image1_brightness.png
> input/image2.png grayscale output/image2_grayscale.png
> input/image3.png blurring output/image3_blurred.png


The output images will be saved to the `output/` directory.

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

## Contributing

Contributions to the project are welcome! If you find a bug or have an idea for a new feature, please submit an issue or a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
