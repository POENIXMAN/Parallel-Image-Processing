#!/bin/bash

# Create the output folder if it doesn't exist
mkdir -p output

# Compile each C file using gcc and move the executable to the root directory
nvcc src/blurring.cu -o blur -lpng
mv blur .

nvcc src/edge_detection.cu -o edge -lpng
mv edge .

nvcc src/sharpening.cu -o sharp -lpng
mv sharp .

nvcc src/grayscale.cu -o gray -lpng
mv gray .

nvcc src/brightness.cu -o bright -lpng
mv bright .

nvcc src/channel_correction.cu -o correction -lpng
mv correction .

# List of input files
input_files=(
   "img/hawk.png"
   "img/image_big_3ch.png"
)

# Loop through the input files and execute each compiled C file with the required arguments
for i in "${input_files[@]}"
do
    ./blur "$i" "output/blur-${i#*/}"
    ./edge "$i" "output/edge-${i#*/}"
    ./sharp "$i" "output/sharp-${i#*/}"
    ./gray "$i" "output/gray-${i#*/}"
    ./bright "$i" "output/bright-${i#*/}"
    ./correction "$i" "output/correction-${i#*/}"
done 
