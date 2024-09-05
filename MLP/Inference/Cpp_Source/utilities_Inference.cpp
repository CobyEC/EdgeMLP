// utilities_Inference.cpp
// Author: Coby Cockrell
// Date: 8/26/2024
// Purpose: The utilities_inference file create the necessary methods for graceful file/weight handling.
// 
#include "utilities_Inference.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

//Function to read float data from a text file
std::vector<std::pair<std::vector<float>, int>> read_float_data(const std::string& file_path) {
    std::vector<std::pair<std::vector<float>, int>> data;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> input;
        std::string token;

        //Read input values
        while (std::getline(ss, token, ',')) {
            try {
                float value = std::stof(token);
                input.push_back(value);
            }
            catch (const std::invalid_argument&) {
                std::cerr << "Warning: Invalid float value '" << token << "' found in input." << std::endl;
            }
            catch (const std::out_of_range&) {
                std::cerr << "Warning: Float value out of range '" << token << "' found in input." << std::endl;
            }
        }

        //Assume the last value is the label
        if (!input.empty()) {
            int label = static_cast<int>(input.back());
            input.pop_back();

            if (label == 0 || label == 1) {
                data.emplace_back(input, label);
            }
            else {
                std::cerr << "Warning: Invalid label " << label << ". Expected 0 or 1." << std::endl;
            }
        }
    }

    file.close();
    return data;
}

//Function to load weights from a file
std::vector<float> load_weights(const std::string& file_path) {
    std::vector<float> weights;
    std::ifstream file(file_path);
    if (file.is_open()) {
        float weight;
        while (file >> weight) {
            weights.push_back(weight);
        }
        file.close();
    }
    else {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
    }
    return weights;
}