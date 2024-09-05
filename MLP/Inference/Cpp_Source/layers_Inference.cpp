// layers_Inference.cpp
// Author: Coby Cockrell
// Date: 8/25/2024
// Purpose: This is the layer implementation file in which each layer of the network is defined for Inference computattion... specifically scrap all training methods


#include "layers_Inference.h"
#include "activate.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

//Layer class implementation
Layer::Layer(uint32_t input_size, uint32_t output_size)
    : input_size(input_size), output_size(output_size) {
    weights.resize(input_size * output_size);
    biases.resize(output_size);
}

//Method Allows to set weights in Inference
void Layer::set_weights(const std::vector<float>& new_weights) {
    if (new_weights.size() != weights.size()) {
        throw std::invalid_argument("Weights size mismatch");
    }
    weights = new_weights;
}

void Layer::set_biases(const std::vector<float>& new_biases) {
    if (new_biases.size() != biases.size()) {
        throw std::invalid_argument("Biases size mismatch");
    }
    biases = new_biases;
}

//InputLayer implementation
InputLayer::InputLayer(uint32_t input_size, uint32_t output_size)
    : Layer(input_size, output_size) {}

void InputLayer::forward(const float* input, float* output) const {
    std::copy(input, input + input_size, output);
}

//HiddenLayer implementation
HiddenLayer::HiddenLayer(uint32_t input_size, uint32_t output_size)
    : Layer(input_size, output_size) {}

void HiddenLayer::forward(const float* input, float* output) const {
    for (uint32_t i = 0; i < output_size; ++i) {
        float sum = std::inner_product(input, input + input_size, weights.begin() + i * input_size, biases[i]);
        output[i] = activate::relu(sum);
    }
}

//OutputLayer implementation
OutputLayer::OutputLayer()
    : Layer(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE) {}

void OutputLayer::forward(const float* input, float* output) const {
    for (uint32_t i = 0; i < output_size; ++i) {
        float sum = std::inner_product(input, input + input_size, weights.begin() + i * input_size, biases[i]);
        output[i] = activate::sigmoid(sum);
    }
}