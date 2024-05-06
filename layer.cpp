// layer.cpp
//Author: Coby Cockrell
//Date: 5/6/2024
//Purpose: This is the layer implimentation file inwhich each layer of the network is defined, and their respective propagation and activation is applied.

#include "layer.h"

//helper function for weight initialization
void initialize_weights(float* weights, uint32_t input_size, uint32_t output_size) {
    static bool initialized = false;
    static uint32_t seed = 42; // Seed for random number 

    if (!initialized) {
        srand(seed);
        initialized = true;
    }

    for (uint32_t i = 0; i < output_size; ++i) {
        for (uint32_t j = 0; j < input_size; ++j) {
            weights[i * input_size + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f; // Random initialization between -0.5 and 0.5
        }
    }
}

//Layer class implementation
Layer::Layer(uint32_t input_size, uint32_t output_size) : input_size(input_size), output_size(output_size) {
    initialize_layer_weights();
}

void Layer::forward(const float* input, float* output) {
    // Placeholder for forward propagation implementation
    // This will be implemented in the derived classes
}

void Layer::initialize_layer_weights() {
    initialize_weights(reinterpret_cast<float*>(weights), input_size, output_size);
    for (uint32_t i = 0; i < output_size; ++i) {
        biases[i] = 0.0f; //initialize biases to zero
    }
}

// InputLayer class implementation
InputLayer::InputLayer() : Layer(INPUT_SIZE, HIDDEN_LAYER1_SIZE) {}

void InputLayer::forward(const float* input, float* output) {
    // Copy input to output (no computation needed for the input layer)
    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        output[i] = input[i];
    }
}

// HiddenLayer class implementation
HiddenLayer::HiddenLayer(uint32_t input_size, uint32_t output_size)
    : Layer(input_size, output_size) {}

void HiddenLayer::forward(const float* input, float* output) {
    // Forward propagation implementation for the hidden layer
    for (uint32_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < input_size; ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum + biases[i]; // Apply activation function (e.g., ReLU) here
    }
}

//outputLayer class implementation
OutputLayer::OutputLayer() : Layer(HIDDEN_LAYER2_SIZE, OUTPUT_SIZE) {}

void OutputLayer::forward(const float* input, float* output) {
    // Forward propagation implementation for the output layer
    for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < HIDDEN_LAYER2_SIZE; ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = 1.0f / (1.0f + exp(-sum - biases[i])); //sigmoid activation function
    }
}
