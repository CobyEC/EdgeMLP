// MLP.cpp
// Author: Coby Cockrell
// Date: 7/21/2024
// Purpose: This is the layer implementation file that defines the functionality of the Multi-Layer Perceptron 
#include "MLP.h"
#include "layers.h"
#include "utilities.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

const std::vector<float>& MLP::get_output() const {
    return output_layer.get_output();
}

// Get functions, one of the downsides of object orientation from classes
Layer& MLP::get_input_layer() {
    return input_layer;
}

const Layer& MLP::get_input_layer() const {
    return input_layer;
}

Layer& MLP::get_hidden_layer1() {
    return hidden_layer1;
}

const Layer& MLP::get_hidden_layer1() const {
    return hidden_layer1;
}

Layer& MLP::get_output_layer() {
    return output_layer;
}

const Layer& MLP::get_output_layer() const {
    return output_layer;
}

// Helper function for input normalization
std::vector<float> MLP::normalize_input(const std::vector<int>& input) {
    // Empty Check
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }

    std::vector<float> normalized(input.size() + 1); // +1 for parity feature
    float max_val = static_cast<float>(*std::max_element(input.begin(), input.end()));

    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = static_cast<float>(input[i]) / max_val;  // Normalize to [0, 1]
    }

    // Add parity feature (0 for even, 1 for odd)
    int last_number = input.back(); // Assuming the last number is the main input number
    normalized[input.size()] = (last_number % 2 == 0) ? 0.0f : 1.0f;

    return normalized;
}

std::vector<float> MLP::get_weights() const {
    std::vector<float> all_weights;

    // Collect weights from input layer
    const auto& input_weights = input_layer.get_weights();
    all_weights.insert(all_weights.end(), input_weights.begin(), input_weights.end());

    // Collect weights from hidden layer
    const auto& hidden_weights = hidden_layer1.get_weights();
    all_weights.insert(all_weights.end(), hidden_weights.begin(), hidden_weights.end());

    // Collect weights from output layer
    const auto& output_weights = output_layer.get_weights();
    all_weights.insert(all_weights.end(), output_weights.begin(), output_weights.end());

    return all_weights;
}

std::vector<float> MLP::get_biases() const {
    std::vector<float> all_biases;

    // Collect biases from input layer
    const auto& input_biases = input_layer.get_biases();
    all_biases.insert(all_biases.end(), input_biases.begin(), input_biases.end());

    // Collect biases from hidden layer
    const auto& hidden_biases = hidden_layer1.get_biases();
    all_biases.insert(all_biases.end(), hidden_biases.begin(), hidden_biases.end());

    // Collect biases from output layer
    const auto& output_biases = output_layer.get_biases();
    all_biases.insert(all_biases.end(), output_biases.begin(), output_biases.end());

    return all_biases;
}

// Initializing member defining the initial layers, and MLP (1 Output Layer) 
MLP::MLP(uint32_t max_input_size)
    : input_layer(max_input_size + 1, HIDDEN_LAYER1_SIZE), // +1 for parity feature
    hidden_layer1(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE),
    output_layer(),
    intermediate(std::max(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE)),
    output(OUTPUT_SIZE) {

    // Size Check
    if (max_input_size < 1 || max_input_size > 9) {
        throw std::invalid_argument("Input size must be between 1 and 9");
    }
    // std::cout << "MLP constructed with max_input_size: " << max_input_size << std::endl;
}

MLP::~MLP() {}

void MLP::forward(const std::vector<float>& input) const {
    // Size Check
    if (input.empty() || input.size() > 10) { // Adjusted size check
        throw std::invalid_argument("Input size must be between 1 and 10");
    }

    // Create padded input
    std::vector<float> padded_input(10, 0.0f);  // Initialize with 10 zeros
    for (size_t i = 0; i < input.size(); ++i) {
        padded_input[i] = static_cast<float>(input[i]);
    }

    // std::cout << "MLP forward start. Input size: " << input.size() << std::endl;

    // Forward pass through layers
    input_layer.forward(padded_input.data(), intermediate.data());
    hidden_layer1.forward(intermediate.data(), intermediate.data());
    output_layer.forward(intermediate.data(), output.data());

    // std::cout << "MLP forward end. Final output: " << output[0] << std::endl;
}

float MLP::predict(const std::vector<float>& input) const {
    // std::cout << "MLP predict start" << std::endl;
    forward(input);
    // std::cout << "MLP predict end. Prediction: " << output[0] << std::endl;
    return output[0];
}
