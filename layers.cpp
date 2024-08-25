// layer.cpp
//Author: Coby Cockrell
//Date: 5/6/2024
//Purpose: This is the layer implementation file in which each layer of the network is defined, and their respective propagation and activation is applied.

#include "layers.h"
#include "activate.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cassert>

//Helper function for weight initialization
void initialize_weights(std::vector<float>& weights, uint32_t input_size, uint32_t output_size) {
    static std::mt19937 rng(42); // Seed for random number generator
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    weights.resize(input_size * output_size);
    for (auto& weight : weights) {
        weight = dist(rng);
    }
    std::cout << "Initialized weights: size = " << weights.size() << std::endl;
}

//Layer class implementation
Layer::Layer(uint32_t input_size, uint32_t output_size)
    : input_size(input_size), output_size(output_size) {

    //Layer positive check
    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("Layer sizes must be positive");
    }
    weights.resize(input_size * output_size);
    biases.resize(output_size);
    initialize_layer_weights();
    std::cout << "Layer constructed: input_size = " << input_size
        << ", output_size = " << output_size
        << ", weights size = " << weights.size()
        << ", biases size = " << biases.size() << std::endl;
}

//Initializing Weights method that allows a random weight initialization per required sizes
void Layer::initialize_layer_weights() {
    initialize_weights(weights, input_size, output_size);
    biases.resize(output_size, 0.0f); // Initialize biases to zero
    //std::cout << "Initialized biases: size = " << biases.size() << std::endl;
}

//InputLayer class implementation
InputLayer::InputLayer(uint32_t input_size, uint32_t output_size) : Layer(INPUT_SIZE, HIDDEN_LAYER1_SIZE) {
    //std::cout << "InputLayer constructed" << std::endl;
}

//InputLayer forward method that  shall take the input, feed the inputt layer, recieve the outputs, and pass them back
void InputLayer::forward(const float* input, float* output) {
    //std::cout << "InputLayer forward: input_size = " << input_size << ", output_size = " << output_size << std::endl;
    input_cache.assign(input, input + input_size);
    if (input_size <= output_size) {
        std::copy(input, input + input_size, output);
        std::fill(output + input_size, output + output_size, 0.0f);
    }
    else {
        std::copy(input, input + output_size, output);
    }
    output_cache.assign(output, output + output_size);
    //std::cout << "InputLayer forward: input[0] = " << input[0] << ", output[0] = " << output[0] << std::endl;
}

void InputLayer::update_weights(float error, float learning_rate) {
    std::cout << "InputLayer update_weights called (no action)" << std::endl;
}

float InputLayer::get_output_derivative() const {
    return 1.0f; //Identity function derivative
}

//HiddenLayer class implementation
HiddenLayer::HiddenLayer(uint32_t input_size, uint32_t output_size)
    : Layer(input_size, output_size) {
    //std::cout << "HiddenLayer constructed: input_size = " << input_size << ", output_size = " << output_size << std::endl;
}

//HiddenLayer forward method that  shall take the output from InputLayer, feed the 1st(And only in this case) hidden layer, recieve the outputs, and pass them back
void HiddenLayer::forward(const float* input, float* output) {

    //Null check
    if (input == nullptr || output == nullptr) {
        throw std::runtime_error("Null pointer in HiddenLayer forward");
    }
    //Layer size check
    if (weights.size() != input_size * output_size || biases.size() != output_size) {
        throw std::runtime_error("Weight or bias size mismatch in HiddenLayer");
    }
    input_cache.assign(input, input + input_size);
    for (uint32_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < input_size; ++j) {
            sum += input[j] * weights.at(i * input_size + j);
        }
        output[i] = activate::relu(activate::clip(sum + biases.at(i), -88.0f, 88.0f));
    }
    output_cache.assign(output, output + output_size);
    //std::cout << "HiddenLayer forward: input[0] = " << input[0] << ", output[0] = " << output[0] << std::endl;
}

//HiddenLayer update weights method that shalll use backpropagation to calculate, and then update the weights for the hidden layer.
void HiddenLayer::update_weights(float error, float learning_rate) {
    assert(input_cache.size() == input_size && "Input cache size mismatch");
    for (uint32_t i = 0; i < output_size; ++i) {
        for (uint32_t j = 0; j < input_size; ++j) {
            weights[i * input_size + j] -= learning_rate * error * input_cache[j];
        }
        biases[i] -= learning_rate * error;
    }
    std::cout << "HiddenLayer update_weights: error = " << error << ", learning_rate = " << learning_rate << std::endl;
}

float HiddenLayer::get_output_derivative() const {
    assert(!output_cache.empty() && "Output cache is empty");
    //Derivative of ReLU
    float sum = 0.0f;
    for (float val : output_cache) {
        sum += (val > 0) ? 1.0f : 0.0f;
    }
    return sum / output_cache.size();
}

//OutputLayer class implementation
OutputLayer::OutputLayer() : Layer(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE) {
    std::cout << "OutputLayer constructed. InputSize: " << input_size << ", OutputSize: " << output_size << std::endl;
    //std::cout << "Weights size: " << weights.size() << ", Biases size: " << biases.size() << std::endl;
}

void OutputLayer::forward(const float* input, float* output) {
    //std::cout << "OutputLayer forward start" << std::endl;
    if (input == nullptr || output == nullptr) {
        throw std::runtime_error("Input or output is null in OutputLayer forward");
    }
    if (weights.size() != input_size * output_size) {
        throw std::runtime_error("Weights size mismatch in OutputLayer");
    }
    if (biases.size() != output_size) {
        throw std::runtime_error("Biases size mismatch in OutputLayer");
    }

    //std::cout << "InputSize: " << input_size << ", OutputSize: " << output_size << std::endl;
    //std::cout << "Weights size: " << weights.size() << ", Biases size: " << biases.size() << std::endl;

    input_cache.assign(input, input + input_size);
    for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < HIDDEN_LAYER1_SIZE; ++j) {
            sum += input[j] * weights[i * HIDDEN_LAYER1_SIZE + j];
        }
        output[i] = activate::sigmoid(sum + biases[i]);
        std::cout << "Output[" << i << "]: " << output[i] << std::endl;
    }
    output_cache.assign(output, output + output_size);
    std::cout << "OutputLayer forward end: input[0] = " << input[0] << ", output[0] = " << output[0] << std::endl;
}

void OutputLayer::update_weights(float error, float learning_rate) {
    assert(input_cache.size() == input_size && "Input cache size mismatch");
    for (uint32_t i = 0; i < output_size; ++i) {
        for (uint32_t j = 0; j < input_size; ++j) {
            weights[i * input_size + j] -= learning_rate * error * input_cache[j];
        }
        biases[i] -= learning_rate * error;
    }
    std::cout << "OutputLayer update_weights: error = " << error << ", learning_rate = " << learning_rate << std::endl;
}

float OutputLayer::get_output_derivative() const {
    assert(!output_cache.empty() && "Output cache is empty");
    // Derivative of Sigmoid
    float output = output_cache[0];
    return output * (1 - output);
}