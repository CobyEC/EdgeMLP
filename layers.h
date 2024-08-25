#pragma once
//layer.h
//Author: Coby Cockrell
//Date: 5/5/2024
#ifndef LAYERS_H
#define LAYERS_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>

//Constants for layer parameters
constexpr uint32_t INPUT_SIZE = 6;
constexpr uint32_t HIDDEN_LAYER1_SIZE = 64;
//constexpr uint32_t HIDDEN_LAYER2_SIZE = 32;
constexpr uint32_t OUTPUT_SIZE = 1;

class Layer {
public:
    Layer(uint32_t input_size, uint32_t output_size);
    virtual ~Layer() =default;

    virtual void forward(const float* input, float* output) = 0;
    virtual void update_weights(float error, float learning_rate) = 0;
    virtual float get_output_derivative() const = 0;

protected:
    uint32_t input_size;
    uint32_t output_size;
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> input_cache;
    std::vector<float> output_cache;

    void initialize_layer_weights();
};

class InputLayer : public Layer {
public:
    InputLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output) override;
    void update_weights(float error, float learning_rate) override;
    float get_output_derivative() const override;
    uint32_t get_input_size() const { return input_size; } 
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output) override;
    void update_weights(float error, float learning_rate) override;
    float get_output_derivative() const override;
};

class OutputLayer : public Layer {
public:
    OutputLayer();
    void forward(const float* input, float* output) override;
    void update_weights(float error, float learning_rate) override;
    float get_output_derivative() const override;
};

#endif