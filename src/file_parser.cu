#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "file_parser.h"

// A function to parse the input file and store the jobs in a vector
std::vector<Job> parse_input_file(const std::string& input_filename) {
    std::vector<Job> jobs;
    std::ifstream input_file(input_filename);
    if (input_file.is_open()) {
        std::string line;
        while (std::getline(input_file, line)) {
            // Split each line into input file, algorithm and output file names
            size_t pos1 = line.find_first_of(" ");
            size_t pos2 = line.find_last_of(" ");
            std::string input_file_name = line.substr(0, pos1);
            std::string algorithm_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
            std::string output_file_name = line.substr(pos2 + 1);
            // Create a job struct and add it to the vector
            Job job = {input_file_name, output_file_name, algorithm_name};
            jobs.push_back(job);
        }
        input_file.close();
    } else {
        std::cerr << "Error: unable to open input file " << input_filename << std::endl;
    }
    return jobs;
}
