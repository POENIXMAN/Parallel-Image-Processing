#!/bin/bash

# Compile each C file using gcc
nvcc -o blur blurring.cu -lpng

# List of input files
input_files=(
   "hawk.png"
)

# Loop through the input files and execute each compiled C file with the required arguments
for i in "${input_files[@]}"
do
    ./blur "$i" "output-${i%.}"
done