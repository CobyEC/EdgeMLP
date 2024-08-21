#pragma once
//layer.h
//Author: Coby Cockrell
//Date: 5/5/2024
#ifndef LAYER_H
#define LAYER_H

#include <cstdint>
#include <cstdlib>

// Constants for layer parameters
constexpr uint32_t INPUT_SIZE = 12;
constexpr uint32_t HIDDEN_LAYER1_SIZE = 64;
constexpr uint32_t HIDDEN_LAYER2_SIZE = 32;
constexpr uint32_t OUTPUT_SIZE = 1;

class Layer {
public:
    Layer(uint32_t input_size, uint32_t output_size);
    virtual ~Layer();

    virtual void forward(const float* input, float* output) = 0;

protected:
    uint32_t input_size;
    uint32_t output_size;
    float weights[HIDDEN_LAYER1_SIZE * INPUT_SIZE];
    float biases[HIDDEN_LAYER1_SIZE];

    void initialize_layer_weights();
};

class InputLayer : public Layer {
public:
    InputLayer();
    void forward(const float* input, float* output) override;
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output) override;

private:
    float hidden_weights[HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE];
    float hidden_biases[HIDDEN_LAYER2_SIZE];
};

class OutputLayer : public Layer {
public:
    OutputLayer();
    void forward(const float* input, float* output) override;

private:
    float output_weights[OUTPUT_SIZE * HIDDEN_LAYER2_SIZE];
    float output_biases[OUTPUT_SIZE];
};

#endif 
