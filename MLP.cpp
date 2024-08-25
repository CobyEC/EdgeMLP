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

//Helper function for input normalization
std::vector<float> normalize_input(const std::vector<int>& input) {

    //Empty Check
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty");
    }
    std::vector<float> normalized(input.size());
    float max_val = static_cast<float>(*std::max_element(input.begin(), input.end()));
    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = static_cast<float>(input[i]) / max_val;  //Normalize to [0, 1]
    }
    return normalized;
}

//Function for gradient clipping
float clip_gradient(float grad, float max_value) {
    return std::max(std::min(grad, max_value), -max_value);
}
//Initializing member defining the initial layers, and MLP (1 Output Layer) 
MLP::MLP(uint32_t max_input_size)
    : input_layer(max_input_size, HIDDEN_LAYER1_SIZE),
    hidden_layer1(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE),
    output_layer(),
    intermediate(std::max(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE)),
    output(OUTPUT_SIZE) {

    //Size Check
    if (max_input_size < 1 || max_input_size > 9) {
        throw std::invalid_argument("Input size must be between 1 and 9");
    }
    std::cout << "MLP constructed with max_input_size: " << max_input_size << std::endl;
}

MLP::~MLP() {}

void MLP::forward(const std::vector<int>& input) {

    //Size Check
    if (input.empty() || input.size() > 9) {
        throw std::invalid_argument("Input size must be between 1 and 9");
    }

    //Create padded input
    std::vector<float> padded_input(9, 0.0f);  //Initialize with 9 zeros
    for (size_t i = 0; i < input.size(); ++i) {
        padded_input[i] = static_cast<float>(input[i]);
    }

    std::cout << "MLP forward start. Input size: " << input.size() << std::endl;

    //Forward pass through layers
    input_layer.forward(padded_input.data(), intermediate.data());
    hidden_layer1.forward(intermediate.data(), intermediate.data());
    output_layer.forward(intermediate.data(), output.data());

    std::cout << "MLP forward end. Final output: " << output[0] << std::endl;
}

float MLP::predict(const std::vector<int>& input) {
    std::cout << "MLP predict start" << std::endl;
    forward(input);
    std::cout << "MLP predict end. Prediction: " << output[0] << std::endl;
    return output[0];
}

//Traininng method that will perform the basic    data load -> forward pass -> loss compputation -> backprop -> weight update
void MLP::train(const std::vector<std::pair<std::vector<int>, int>>& training_data, int epochs, float learning_rate) {

    //Epoch positive check
    if (epochs <= 0) {
        throw std::invalid_argument("Number of epochs must be positive");
    }
    //Learning rate range Check
    if (learning_rate <= 0.0f || learning_rate >= 1.0f) {
        throw std::invalid_argument("Learning rate must be between 0 and 1");
    }

    std::vector<float> errors(3);  //Reuse this vector for all error calculations
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct_predictions = 0;

        for (const auto& data_pair : training_data) {
            const std::vector<int>& input = data_pair.first;
            int target = data_pair.second;

            if (input.empty() || input.size() > 9) {
                throw std::invalid_argument("Training data input size must be between 1 and 9");
            }
            if (target != 0 && target != 1) {
                throw std::invalid_argument("Target must be 0 or 1");
            }

            //Forward pass
            forward(input);

            //Compute loss
            float loss = 0.5f * std::pow(output[0] - target, 2);
            total_loss += loss;

            //Check if prediction is correct
            if ((output[0] > 0.5f && target == 1) || (output[0] <= 0.5f && target == 0)) {
                ++correct_predictions;
            }

            //Backpropagation
            errors[0] = clip_gradient(output[0] - target, 1.0f);
            errors[1] = errors[0] * output_layer.get_output_derivative();
            errors[2] = errors[1] * hidden_layer1.get_output_derivative();

            //Update weights
            output_layer.update_weights(errors[0], learning_rate);
            hidden_layer1.update_weights(errors[1], learning_rate);
        }

        //Print epoch statistics
        float accuracy = static_cast<float>(correct_predictions) / training_data.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss / training_data.size()
            << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
    }
}