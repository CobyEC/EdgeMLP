//layer.h
//Author: Coby Cockrell
//Date: 5/5/2024
#ifndef LAYER_H
#define LAYER_H

#include <cstdint>
#include <cstdlib>

//constants for layer parameters
constexpr uint32_t INPUT_SIZE = 12;
constexpr uint32_t HIDDEN_LAYER1_SIZE = 64;
constexpr uint32_t HIDDEN_LAYER2_SIZE = 32;
constexpr uint32_t OUTPUT_SIZE = 1;

class Layer {
public:
    //constructor
    Layer(uint32_t input_size, uint32_t output_size);

    //forward propagation function
    void forward(const float* input, float* output);

protected:
    uint32_t input_size;
    uint32_t output_size;
    float weights[HIDDEN_LAYER2_SIZE][HIDDEN_LAYER1_SIZE];
    float biases[HIDDEN_LAYER2_SIZE];

    // Helper functions
    void initialize_layer_weights();
};

class InputLayer : public Layer {
public:
    InputLayer();
    void forward(const float* input, float* output);
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output);
};

class OutputLayer : public Layer {
public:
    OutputLayer();
    void forward(const float* input, float* output);
};

#endif // LAYER_H