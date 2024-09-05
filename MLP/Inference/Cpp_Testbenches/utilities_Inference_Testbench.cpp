// utilities_Inference_Testbench.cpp
// Author: Coby Cockrell
// Date: 8/26/2024
// Purpose: This file contains a main function to test the utility functions in utilities_Inference

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include "utilities_Inference.h"
#include "test.txt"


//Function to test the read_float_data function
void test_read_float_data() {
    std::cout << "Testing read_float_data function..." << std::endl;
    std::vector<std::pair<std::vector<float>, int>> data = read_float_data("train.txt");

    if (data.empty()) {
        std::cerr << "Error: No data was read from 'train.txt'. "
            << "Please ensure the file exists and is not empty." << std::endl;
        std::cerr << "Make sure 'train.txt' is in the same directory as the executable." << std::endl;
        return;
    }

    std::cout << "Successfully read " << data.size() << " samples from file." << std::endl;

    //Check a few samples if we have enough data
    if (data.size() >= 2) {
        std::cout << "First sample: Input = ";
        for (float value : data[0].first) {
            std::cout << value << " ";
        }
        std::cout << ", Label = " << data[0].second << std::endl;

        std::cout << "Second sample: Input = ";
        for (float value : data[1].first) {
            std::cout << value << " ";
        }
        std::cout << ", Label = " << data[1].second << std::endl;
    }
    else {
        std::cout << "Warning: Not enough samples to display." << std::endl;
    }

    std::cout << "read_float_data test completed." << std::endl << std::endl;
}

//Function to test the load_weights function
void test_load_weights() {
    std::cout << "Testing load_weights function..." << std::endl;
    std::vector<float> weights = load_weights("weights.txt");

    if (weights.empty()) {
        std::cerr << "Error: No weights were loaded from 'weights.txt'. "
            << "Please ensure the file exists and is not empty." << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << weights.size() << " weights from file." << std::endl;

    //Check a few weights if we have enough data
    std::cout << "Sample weights: ";
    for (size_t i = 0; i < std::min(weights.size(), size_t(5)); ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "load_weights test completed." << std::endl << std::endl;
}

int main() {
    test_read_float_data();
    test_load_weights();

    std::cout << "All tests completed." << std::endl;
    return 0;
}