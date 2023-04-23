#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_processing.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s jobs.txt\n", argv[0]);
        return 1;
    }

    // Open jobs file
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("Error: Failed to open file %s\n", argv[1]);
        return 1;
    }

    char input_file[256];
    char algorithm[256];
    char output_file[256];

    // Read jobs and process images
    while (fscanf(fp, "%s %s %s", input_file, algorithm, output_file) == 3) {
        printf("Processing job: %s %s %s\n", input_file, algorithm, output_file);

        // Load input image
        Image input_image;
        if (!load_image(input_file, &input_image)) {
            printf("Error: Failed to load input image %s\n", input_file);
            continue;
        }

        // Apply image processing algorithm
        if (strcmp(algorithm, "brightness") == 0) {
            brightness(&input_image, 50);
        } else if (strcmp(algorithm, "grayscale") == 0) {
            grayscale_luminosity(&input_image);
        } else if (strcmp(algorithm, "channel_correction") == 0) {
            channel_correction(&input_image);
        } else if (strcmp(algorithm, "blurring") == 0) {
            blurring(&input_image);
        } else if (strcmp(algorithm, "sharpening") == 0) {
            sharpening(&input_image);
        } else if (strcmp(algorithm, "edge_detection") == 0) {
            edge_detection(&input_image);
        } else {
            printf("Error: Unknown algorithm %s\n", algorithm);
            continue;
        }

        // Save output image
        if (!save_image(output_file, &input_image)) {
            printf("Error: Failed to save output image %s\n", output_file);
        }

        // Free memory
        free_image(&input_image);
    }

    // Close jobs file
    fclose(fp);

    return 0;
}
