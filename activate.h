// activation.h
// Author: Coby Cockrell
// Date: 5/7/2024
// Purpose: This header file declares activation functions for the neural network.

#ifndef ACTIVATE_H
#define ACTIVATE_H

#include <cmath>

namespace activate {
    //ReLU activation function
    float relu(float x);

    //Leaky ReLU activation function
    float leaky_relu(float x, float alpha = 0.01f);

    //Sigmoid activation function
    float sigmoid(float x);

    //Tanh activation function
    float tanh(float x);

    //Softmax activation function (for output layer)
    void softmax(float* input, float* output, int size);
}

#endif