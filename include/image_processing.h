#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef unsigned char byte;

typedef struct Image {
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
} Image;

int load_image(const char *filename, Image *image);
int save_image(const char *filename, Image *image);
void free_image(Image *image);
void grayscale_luminosity(Image *image) {
    byte *input = image->data;
    byte *output = (byte *)malloc(image->width * image->height * sizeof(byte));
    grayscale(input, output, image->width, image->height);
    free(input);
    image->data = output;
    image->channels = 1;
}

void brightness(Image *image, int value);
void channel_correction(Image *image);
void blurring(Image *image);
void sharpening(Image *image);
void edge_detection(Image *image);

#endif /* IMAGE_PROCESSING_H */


