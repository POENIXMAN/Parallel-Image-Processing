
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>

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

__global__ void SobelKernel(png_byte *d_P, int height, int width)
{
    // Define the Sobel filter kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Calculate the global index of the current thread
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = tid_y * width + tid_x;

    // Check if the thread is within the image boundaries
    if (tid_x >= width || tid_y >= height)
    {
        return;
    }

    // Calculate the pixel values in the neighborhood of the current pixel
    int sumx = 0, sumy = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int y = tid_y + i;
            int x = tid_x + j;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                int index = (y * width + x) * 3;
                sumx += Gx[i + 1][j + 1] * d_P[index];
                sumy += Gy[i + 1][j + 1] * d_P[index];
            }
        }
    }

    // Calculate the gradient magnitude of the current pixel
    int magnitude = abs(sumx) + abs(sumy);

    // Set the color of the pixel based on the gradient magnitude
    png_byte gray = (png_byte)magnitude;
    d_P[tid * 3] = gray;
    d_P[tid * 3 + 1] = gray;
    d_P[tid * 3 + 2] = gray;
}

void process_on_host(PNG_RAW *png_raw)
{
    long long start = timeInMilliseconds();
    int width = png_raw->width;
    int height = png_raw->height;

    // Create a temporary buffer to store the luminance values
    int *luminance = (int *)malloc(sizeof(int) * width * height);

    // Compute the luminance values for each pixel
    for (int i = 0; i < width * height; i++)
    {
        int r = png_raw->buf[i * 3];
        int g = png_raw->buf[i * 3 + 1];
        int b = png_raw->buf[i * 3 + 2];

        int luminance_value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luminance[i] = luminance_value;
    }

    // Compute the edge detection values for each pixel
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int i = y * width + x;

            int gx = -luminance[i - width - 1] - 2 * luminance[i - 1] - luminance[i + width - 1] +
                     luminance[i - width + 1] + 2 * luminance[i + 1] + luminance[i + width + 1];
            int gy = -luminance[i - width - 1] - 2 * luminance[i - width] - luminance[i - width + 1] +
                     luminance[i + width - 1] + 2 * luminance[i + width] + luminance[i + width + 1];

            int edge_value = sqrt(gx * gx + gy * gy);

            // Normalize the edge value to fit in the range [0, 255]
            edge_value = (int)(255.0 * edge_value / (sqrt(2.0) * 255.0));

            png_raw->buf[i * 3] = (png_byte)edge_value;
            png_raw->buf[i * 3 + 1] = (png_byte)edge_value;
            png_raw->buf[i * 3 + 2] = (png_byte)edge_value;
        }
    }

    free(luminance);

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

    SobelKernel<<<DimGrid, DimBlock>>>(d_P, m, n);

    cudaMemcpy(png_raw->buf, d_P, m * n * pixel_size, cudaMemcpyDeviceToHost);

    long long end = timeInMilliseconds();

    printf("timing on Device is %lld millis\n", end - start);
}

int main(int argc, char **argv)
{
    int on_host = 1;

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
