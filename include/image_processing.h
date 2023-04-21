#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

struct uchar3 {
    unsigned char x, y, z;
};

void brightness(const uchar3* input, uchar3* output, int width, int height, int brightness_value);
void grayscale(const uchar3* input, unsigned char* output, int width, int height);
void channel_correction(const uchar3* input, uchar3* output, int width, int height, float red_scale, float green_scale, float blue_scale);
void box_blur(const uchar3* input, uchar3* output, int width, int height, int kernel_size);
void edge_detection(const uchar3* input, uchar3* output, int width, int height);
void sharpening(const uchar3* input, uchar3* output, int width, int height);

#endif
