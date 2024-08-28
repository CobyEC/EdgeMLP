// activation.h
// Author: Coby Cockrell
// Date: 5/7/2024
// Purpose: This header file declares activation functions for the neural network.

#ifndef ACTIVATE_H
#define ACTIVATE_H

#include <cmath>
#include <algorithm>

namespace activate {
    //ReLU activation function
    inline float relu(float x) {
        return std::max(0.0f, x);
    }

    //Leaky ReLU activation function
    float leaky_relu(float x, float alpha = 0.01f);

    //Sigmoid activation function
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float clip(float x, float min, float max) {
        return std::max(min, std::min(x, max));
    }

    //Tanh activation function
    float tanh(float x);

    //Softmax activation function (for output layer)
    void softmax(float* input, float* output, int size);
}

#endif