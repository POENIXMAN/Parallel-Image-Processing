#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <sys/time.h>
#include <cuda_runtime.h>

typedef struct
{
    int height;
    int width;
    int pixel_size;
    png_infop info_ptr;
    png_byte *buf;
} PNG_RAW;

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

PNG_RAW *read_png(char *file_name)
{
    PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));

    FILE *fp = fopen(file_name, "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int pixel_size = png_get_rowbytes(png_ptr, info_ptr) / width;
    png_raw->width = width;
    png_raw->height = height;
    png_raw->pixel_size = pixel_size;
    png_raw->buf = (png_byte *)malloc(width * height * pixel_size * sizeof(png_byte));
    png_raw->info_ptr = info_ptr;
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            png_raw->buf[k++] = row_pointers[i][j];
        }
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
    return png_raw;
}

void write_png(char *file_name, PNG_RAW *png_raw)
{
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_infop info_ptr = png_raw->info_ptr;
    int width = png_raw->width;
    int height = png_raw->height;
    int pixel_size = png_raw->pixel_size;
    png_bytepp row_pointers;
    row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (int i = 0; i < height; i++)
        row_pointers[i] = (png_bytep)malloc(width * pixel_size);
    int k = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width * pixel_size; j++)
        {
            row_pointers[i][j] = png_raw->buf[k++];
        }

    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    for (int i = 0; i < height; i++)
        free(row_pointers[i]);
    free(row_pointers);
    fclose(fp);
}

__global__ void PictureKernel(png_byte *d_P, int height, int width) {
    // Calculate the row # of the d_P element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_P element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread computes one element of d_P if in range
    if ((Row < height) && (Col < width)) {
        float r = d_P[(Row * width + Col) * 3 + 0] / 255.0f;
        float g = d_P[(Row * width + Col) * 3 + 1] / 255.0f;
        float b = d_P[(Row * width + Col) * 3 + 2] / 255.0f;
        r = fminf(fmaxf(r * 2, 0.0f), 1.0f);
        g = fminf(fmaxf(g * 2, 0.0f), 1.0f);
        b = fminf(fmaxf(b * 2, 0.0f), 1.0f);
        d_P[(Row * width + Col) * 3 + 0] = static_cast<png_byte>(r * 255.0f);
        d_P[(Row * width + Col) * 3 + 1] = static_cast<png_byte>(g * 255.0f);
        d_P[(Row * width + Col) * 3 + 2] = static_cast<png_byte>(b * 255.0f);
    }
}


void process_on_host(PNG_RAW *png_raw)
{
    long long start = timeInMilliseconds();
    for (int i = 0; i < png_raw->width * png_raw->height; i++)
    {
        int luminance_value = 0.2126 * png_raw->buf[i * 3] + 0.7152 * png_raw->buf[i * 3 + 1] + 0.0722 * png_raw->buf[i * 3 + 2];
        png_raw->buf[i * 3] = (png_byte)luminance_value;
        png_raw->buf[i * 3 + 1] = (png_byte)luminance_value;
        png_raw->buf[i * 3 + 2] = (png_byte)luminance_value;
    }
    long long end = timeInMilliseconds();
    printf("timing on host is %lld millis\n", end - start);
}

void process_on_device(PNG_RAW *png_raw)
{

    // assume that the picture is m × n,
    // m pixels in y dimension and n pixels in x dimension
    // input d_Pin has been allocated on and copied to device
    // output d_Pout has been allocated on device
    int m = png_raw->height;
    int n = png_raw->width;
    int pixel_size = png_raw->pixel_size;

    dim3 DimGrid((n - 1) / 16 + 1, (m - 1) / 16 + 1, 1);
    dim3 DimBlock(16, 16, 1);

    png_byte *d_P;
    cudaError_t err;

    long long start = timeInMilliseconds();

    err = cudaMalloc((void **)&d_P, m * n * pixel_size * sizeof(png_byte));
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_P, png_raw->buf, m * n * pixel_size, cudaMemcpyHostToDevice);

    PictureKernel<<<DimGrid, DimBlock>>>(d_P, m, n);

    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size, cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("timing on Device is %lld millis\n", end - start);
}

int main(int argc, char **argv)
{
    int on_host = 2;

    if (argv[3] != NULL && strcmp(argv[3], "-d") == 0)
        on_host = 0;

    PNG_RAW *png_raw = read_png(argv[1]);
    if (png_raw->pixel_size != 3)
    {
        printf("Error, png file must be on 3 Bytes per pixel\n");
        exit(0);
    }
    else
        printf("RGB Processing for Image of %d x %d pixels\n", png_raw->width, png_raw->height);

    if (on_host == 1)
        process_on_host(png_raw);
    else
        process_on_device(png_raw);

    write_png(argv[2], png_raw);

    printf("Processing finished \n");
}