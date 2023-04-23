#include "grayscale.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "image_processing.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void grayscaleKernel(const byte* inputImage, byte* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        byte red = inputImage[3 * index];
        byte green = inputImage[3 * index + 1];
        byte blue = inputImage[3 * index + 2];

        byte gray = (byte)(0.21f * red + 0.71f * green + 0.07f * blue);

        outputImage[index] = gray;
    }
}

void grayscale(const byte* inputImage, byte* outputImage, int width, int height) {
    // Calculate the total number of pixels in the image
    int numPixels = width * height;

    // Allocate memory on the device for the input and output images
    byte* d_inputImage;
    byte* d_outputImage;
    cudaMalloc(&d_inputImage, numPixels * 3 * sizeof(byte));
    cudaMalloc(&d_outputImage, numPixels * sizeof(byte));

    // Copy the input image from the host to the device
    cudaMemcpy(d_inputImage, inputImage, numPixels * 3 * sizeof(byte), cudaMemcpyHostToDevice);

    // Define the thread block and grid sizes
    int blockSize = 256;
    dim3 gridDim((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);
    dim3 blockDim(blockSize, blockSize);

    // Launch the kernel
    grayscaleKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);

    // Copy the output image from the device to the host
    cudaMemcpy(outputImage, d_outputImage, numPixels * sizeof(byte), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
