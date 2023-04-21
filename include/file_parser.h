#ifndef FILE_PARSER_H
#define FILE_PARSER_H

#include <string>
#include <vector>

struct Job {
    std::string input_filename;
    std::string output_filename;
    std::string algorithm_name;
};

std::vector<Job> parse_input_file(const std::string& input_file_path);

#endif /* FILE_PARSER_H */
