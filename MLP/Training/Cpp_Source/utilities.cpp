// utils.cpp
// Author: Coby Cockrell
// Date: 5/1/2024
// Purpose: This file contains the implementation of utility functions for reading data, initializing weights, and saving/loading weights.

#include "utilities.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>


std::vector<std::pair<std::vector<float>, int>> read_float_data(const std::string& file_path) {
    std::vector<std::pair<std::vector<float>, int>> data;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return data;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::vector<float> input;
        std::string token;

        // Read and print the entire line
        std::cout << "Line " << line_number << ": " << line << std::endl;

        // Read input values
        while (std::getline(ss, token, ',')) {
            try {
                float value = std::stof(token);
                input.push_back(value);
            }
            catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Invalid float value '" << token << "' found in input at line " << line_number << std::endl;
            }
            catch (const std::out_of_range& e) {
                std::cerr << "Warning: Float value out of range '" << token << "' found in input at line " << line_number << std::endl;
            }
        }

        // Check if input is not empty
        if (input.empty()) {
            std::cerr << "Warning: Empty input at line " << line_number << std::endl;
            continue;
        }

        // Assume the last value is the label
        int label = static_cast<int>(input.back());
        input.pop_back();

        if (label != 0 && label != 1) {
            std::cerr << "Warning: Invalid label " << label << " at line " << line_number << ". Expected 0 or 1." << std::endl;
            continue;
        }

        data.emplace_back(input, label);
    }

    file.close();

    std::cout << "Successfully read " << data.size() << " samples from " << file_path << std::endl;

    return data;
}


std::vector<std::pair<std::vector<int>, int>> read_data(const std::string& file_path) {
    std::vector<std::pair<std::vector<int>, int>> data;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return data;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::vector<int> input;
        int label;
        std::string token;

        //Read input values
        if (std::getline(ss, token, ',')) {
            for (char c : token) {
                if (std::isdigit(c)) {
                    input.push_back(c - '0');
                }
                else {
                    std::cerr << "Warning: Non-digit character '" << c << "' found in input at line " << line_number << std::endl;
                }
            }
        }
        else {
            std::cerr << "Error: Invalid format at line " << line_number << std::endl;
            continue;
        }

        //Read label
        if (ss >> label) {
            if (label != 0 && label != 1) {
                std::cerr << "Warning: Invalid label " << label << " at line " << line_number << ". Expected 0 or 1." << std::endl;
                continue;
            }
        }
        else {
            std::cerr << "Error: Unable to read label at line " << line_number << std::endl;
            continue;
        }

        //Check if input is not empty
        if (input.empty()) {
            std::cerr << "Warning: Empty input at line " << line_number << std::endl;
            continue;
        }

        data.emplace_back(input, label);
    }

    file.close();

    std::cout << "Successfully read " << data.size() << " samples from " << file_path << std::endl;

    return data;
}

void initialize_weights(std::vector<float>& weights, int num_weights, std::vector<float>& biases, int num_biases) {
    static std::mt19937 rng(std::time(nullptr));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    weights.resize(num_weights);
    for (int i = 0; i < num_weights; ++i) {
        weights[i] = dist(rng);
    }

    biases.resize(num_biases);
    for (int i = 0; i < num_biases; ++i) {
        biases[i] = dist(rng);
    }
}

void save_weights(const std::vector<float>& weights, const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (float weight : weights) {
            file << weight << "\n";
            //std::printf("Weight init...\n");
        }
        file.close();
    }
}

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
    return weights;
}

//Function for gradient clipping
float clip_gradient(float grad, float max_value) {
    return std::max(std::min(grad, max_value), -max_value);
}