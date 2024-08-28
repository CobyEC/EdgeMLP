// utils_Testbench.cpp
// Author: Coby Cockrell
// Date: 5/1/2024
// Purpose: This file contains a main function to test the utility functions in utils.cpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>
#include "utilities.h"

void test_read_data() {
    std::cout << "Testing read_data function..." << std::endl;
    std::vector<std::pair<std::vector<int>, int>> data = read_data("train.txt");

    if (data.empty()) {
        std::cerr << "Error: No data was read from 'train.txt'. "
            << "Please ensure the file exists and is not empty." << std::endl;
        std::cerr << "Make sure 'train.txt' is in the same directory as the executable." << std::endl;
        return;
    }

    std::cout << "Successfully read " << data.size() << " samples from file." << std::endl;

    // Check a few samples if we have enough data
    if (data.size() >= 2) {
        if (data[0].first.size() != 2) {
            std::cerr << "Warning: Expected input to have 2 digits, but got "
                << data[0].first.size() << " digits." << std::endl;
        }
        std::cout << "First sample: Input = ";
        for (int digit : data[0].first) {
            std::cout << digit;
        }
        std::cout << ", Label = " << data[0].second << std::endl;

        std::cout << "Second sample: Input = ";
        for (int digit : data[1].first) {
            std::cout << digit;
        }
        std::cout << ", Label = " << data[1].second << std::endl;
    }
    else {
        std::cout << "Warning: Not enough samples to display." << std::endl;
    }

    std::cout << "read_data test completed." << std::endl << std::endl;
}

bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

void test_initialize_and_save_weights() {
    std::cout << "Testing initialize_weights and save_weights functions..." << std::endl;
    int num_weights = 2464;  // Use the actual number of weights in your network
    int num_biases = 97;     // Use the actual number of biases in your network
    std::vector<float> weights, biases;

    bool weights_exist = file_exists("weights.txt");
    bool biases_exist = file_exists("biases.txt");

    if (weights_exist && biases_exist) {
        std::cout << "Existing weight and bias files found. Loading..." << std::endl;
        weights = load_weights("weights.txt");
        biases = load_weights("biases.txt");

        if (weights.size() != num_weights || biases.size() != num_biases) {
            std::cerr << "Error: Loaded weights or biases do not match expected sizes." << std::endl;
            std::cerr << "Expected " << num_weights << " weights, loaded " << weights.size() << std::endl;
            std::cerr << "Expected " << num_biases << " biases, loaded " << biases.size() << std::endl;
            std::cerr << "Reinitializing weights and biases..." << std::endl;
            initialize_weights(weights, num_weights, biases, num_biases);
        }
    }
    else {
        std::cout << "Weight or bias files not found. Initializing new weights and biases..." << std::endl;
        initialize_weights(weights, num_weights, biases, num_biases);
    }

    // Check if weights and biases are within expected range
    bool weights_in_range = true;
    bool biases_in_range = true;
    for (float w : weights) {
        if (w < -1.0f || w > 1.0f) {
            weights_in_range = false;
            break;
        }
    }
    for (float b : biases) {
        if (b < -1.0f || b > 1.0f) {
            biases_in_range = false;
            break;
        }
    }

    if (!weights_in_range) {
        std::cerr << "Warning: Some weights are out of the expected range [-1, 1]" << std::endl;
    }
    if (!biases_in_range) {
        std::cerr << "Warning: Some biases are out of the expected range [-1, 1]" << std::endl;
    }

    // Save weights and biases
    save_weights(weights, "weights.txt");
    save_weights(biases, "biases.txt");

    std::cout << "Weights saved to 'weights.txt' and biases saved to 'biases.txt'" << std::endl;
    std::cout << "initialize_weights and save_weights tests completed." << std::endl << std::endl;
}

void test_load_weights() {
    // ... (keep this function as is)
}

int main() {
    test_read_data();
    test_initialize_and_save_weights();
    test_load_weights();

    std::cout << "All tests completed." << std::endl;
    return 0;
}