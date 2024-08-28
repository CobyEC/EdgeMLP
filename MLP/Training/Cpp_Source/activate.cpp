// activation.cpp
// Author: Coby Cockrell
// Date: 5/7/2024
// Purpose: This file implements various activation functions for the neural network.

#include "activate.h"
#include <algorithm>


float activate::leaky_relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}


float activate::tanh(float x) {
    return std::tanh(x);
}

void activate::softmax(float* input, float* output, int size) {
    float max_val = input[0];
    float sum = 0.0f;

    //Find maximum value for numerical stability
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    //Compute exponentials and sum
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    //Normalize
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}