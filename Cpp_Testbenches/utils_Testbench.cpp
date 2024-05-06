// utils_Testbench.cpp
// Author: Coby Cockrell
// Date: 5/1/2024
// Purpose: This file contains a main function to test the utility functions in utils.cpp

#include "utils.h"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    int num_weights = 51040;
    int num_biases = 160;

    std::vector<std::pair<std::vector<int>, int>> data = read_data("data.txt");
    std::cout << "Read data from file:" << std::endl;
    for (const auto& pair : data) {
        std::cout << "Input: ";
        for (int digit : pair.first) {
            std::cout << digit;
        }
        std::cout << ", Label: " << pair.second << std::endl;
    }

    // Test read_data function
    std::vector<float> weights;
    std::vector<float> biases;
    initialize_weights(weights, num_weights, biases, num_biases);
    std::cout << "Initialized weights: ";

    for (int i = 0; i < num_weights ; i++) {
        if (i % 1000 == 0) {
            std::cout << "i" << i << "  : Weight :" << weights[i] << std::endl;
        }
    }
    // Test save_weights and load_weights functions
    save_weights(weights, "weights.txt");
    save_weights(biases, "biases.txt");

    std::vector<float> loaded_weights = load_weights("weights.txt");
    std::cout << "Loaded weights: ";

    for (int i = 0; i < num_weights; i++) {
        if (i % 1000 == 0) {
            std::cout << "i" << i << "  : Weight :" << loaded_weights[i] << std::endl;
        }
    }
    std::cout << std::endl;

    for (int i = 0; i < num_weights; i++) {
        if (i % 1000 == 0) {
            double diff = std::abs((double)weights[i] - (double)loaded_weights[i]);
            if (diff <= 0.00001){
                std::cout << (i/ (double) num_weights)*100 << "%   ===>  Weight :" << weights[i] << " : Loaded Weight : " << loaded_weights[i] << "  --- HIT" << std::endl;
            }
            else {
                std::cout << (i / (double) num_weights) * 100 << "%   ===>  Weight :" << weights[i] << " : Loaded Weight : " << loaded_weights[i] << "  --- MISS!!!" << std::endl;
            }
        }
    }

    return 0;
}