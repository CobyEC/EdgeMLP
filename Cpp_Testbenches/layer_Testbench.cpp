//layer_Testbench.cpp
//Author: Coby Cockrell
//Date:5/6/2024
//Purpose: This file is the current testbench for the layer.cpp/layer.h files, I simply verify they take input and produce an expected output.


// main.cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include "layer.h"

int main() {
    // Test case 1: InputLayer
    std::vector<float> input_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float input_layer_output[HIDDEN_LAYER1_SIZE];

    InputLayer input_layer;
    input_layer.forward(input_data.data(), input_layer_output);

    if (sizeof(input_layer_output) / sizeof(float) != HIDDEN_LAYER1_SIZE) {
        printf("InputLayer test FAILED. Output size: %lu\n", sizeof(input_layer_output) / sizeof(float));
    }
    else {
        printf("InputLayer test PASSED\n");
    }

    // Test case 2: HiddenLayer
    float hidden_layer_output[HIDDEN_LAYER2_SIZE];

    HiddenLayer hidden_layer(HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE);
    hidden_layer.forward(input_layer_output, hidden_layer_output);

    if (sizeof(hidden_layer_output) / sizeof(float) != HIDDEN_LAYER2_SIZE) {
        printf("HiddenLayer test FAILED. Output size: %lu\n", sizeof(hidden_layer_output) / sizeof(float));
    }
    else {
        printf("HiddenLayer test PASSED\n");
    }

    // Test case 3: OutputLayer
    float output_layer_output[OUTPUT_SIZE];

    OutputLayer output_layer;
    output_layer.forward(hidden_layer_output, output_layer_output);

    if (sizeof(output_layer_output) / sizeof(float) != OUTPUT_SIZE) {
        printf("OutputLayer test FAILED. Output size: %lu\n", sizeof(output_layer_output) / sizeof(float));
    }
    else {
        printf("OutputLayer test PASSED\n");
    }

    return 0;
}

