#include "grayscale.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "image_processing.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>
#include <png.h>

__global__ void grayscale_kernel(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = image[idx];
        unsigned char g = image[idx + 1];
        unsigned char b = image[idx + 2];
        unsigned char gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        image[idx] = gray;
        image[idx + 1] = gray;
        image[idx + 2] = gray;
    }
}

void read_png_file(const char* file_name, int* width, int* height, png_bytep* row_pointers) {
    FILE* fp = fopen(file_name, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(1);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        std::cerr << "Error: Cannot create read struct" << std::endl;
        fclose(fp);
        exit(1);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error: Cannot create info struct" << std::endl;
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        exit(1);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error: Cannot set jump buffer" << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(1);
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);

    if (color_type != PNG_COLOR_TYPE_RGB && color_type != PNG_COLOR_TYPE_RGBA) {
        std::cerr << "Error: Invalid color type" << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(1);
    }

    if (bit_depth != 8) {
        std::cerr << "Error: Unsupported bit depth" << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(1);
    }

    png_read_update_info(png_ptr, info_ptr);

    *row_pointers = (png_bytep) malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++) {
        (*row_pointers)[y] = (png_byte) malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, *row_pointers);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
}

void write_png_file(const char* file_name, int width, int height, png_bytep* row_pointers) {
    FILE* fp = fopen(file_name, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << file_name << std::endl;
        exit(1);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        std::cerr << "Error: Cannot create write struct" << std::endl;
        fclose(fp);
        exit(1);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error: Cannot create info struct" << std::endl;
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        exit(1);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error: Cannot set jump buffer" << std::endl;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        exit(1);
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    for (int y = 0; y < height; y++) {
        png_write_row(png_ptr, row_pointers[y]);
    }

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input.png>" << std::endl;
        return 1;
    }

    const char *input_filename = argv[1];
    int width, height;
    png_bytep *row_pointers;
    read_png_file(input_filename, &width, &height, &row_pointers);

    int size = width * height * 3 * sizeof(unsigned char);
    unsigned char *d_image;
    cudaMalloc((void**) &d_image, size);
    cudaMemcpy(d_image, row_pointers[0], size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    grayscale_kernel<<<grid_size, block_size>>>(d_image, width, height);

    cudaMemcpy(row_pointers[0], d_image, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    const char *output_filename = "output.png";
    write_png_file(output_filename, width, height, row_pointers);

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    return 0;
}

