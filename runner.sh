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
)

# Loop through the input files and execute each compiled C file with the required arguments
for i in "${input_files[@]}"
do
    ./blur "$i" "output/output-${i#*/}"
    ./edge "$i" "output/output-${i#*/}"
    ./sharp "$i" "output/output-${i#*/}"
    ./gray "$i" "output/output-${i#*/}"
    ./bright "$i" "output/output-${i#*/}"
    ./correction "$i" "output/output-${i#*/}"
done 
