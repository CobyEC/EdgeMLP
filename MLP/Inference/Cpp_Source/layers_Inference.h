#pragma once
// layers.h
// Author: Coby Cockrell
// Date: 8/24/2024

#ifndef LAYERS_INFERENCE_H
#define LAYERS_INFERENCE_H

#include <cstdint>
#include <vector>

//Constants for layer parameters
constexpr uint32_t INPUT_SIZE = 9;
constexpr uint32_t HIDDEN_LAYER1_SIZE = 64;
constexpr uint32_t OUTPUT_SIZE = 1;

class Layer {
public:
    Layer(uint32_t input_size, uint32_t output_size);
    virtual ~Layer() = default;

    virtual void forward(const float* input, float* output) const = 0;
    const std::vector<float>& get_weights() const { return weights; }
    const std::vector<float>& get_biases() const { return biases; }

    void set_weights(const std::vector<float>& new_weights);
    void set_biases(const std::vector<float>& new_biases);

protected:
    uint32_t input_size;
    uint32_t output_size;
    std::vector<float> weights;
    std::vector<float> biases;
};

class InputLayer : public Layer {
public:
    InputLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output) const override;
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(uint32_t input_size, uint32_t output_size);
    void forward(const float* input, float* output) const override;
};

class OutputLayer : public Layer {
public:
    OutputLayer();
    void forward(const float* input, float* output) const override;
};

#endif