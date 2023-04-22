#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include "image_processing.h"

/**
 * Convert an image to grayscale using the luminosity method.
 * @param input The input image.
 * @param output The output image.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 */
void grayscale_luminosity(const byte* input, byte* output, int width, int height);

#endif // GRAYSCALE_H
