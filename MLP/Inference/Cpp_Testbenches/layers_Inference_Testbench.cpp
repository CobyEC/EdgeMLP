// layers_Testbench.cpp
// Author: Coby Cockrell
// Date: 5/9/2024
// Purpose: This file is the testbench for the layer.cpp/layer.h files, verifying they take input and produce expected output.

#include "layers_Inference.h"
#include <iostream>
#include <vector>
#include <cassert>

//Method to test the Input Layer and ensure proper data flow
void test_input_layer() {
    std::cout << "Testing InputLayer..." << std::endl;
    InputLayer input_layer(INPUT_SIZE, HIDDEN_LAYER1_SIZE);

    std::vector<float> input(INPUT_SIZE, 1.0f);
    std::vector<float> output(HIDDEN_LAYER1_SIZE, 0.0f);

    input_layer.forward(input.data(), output.data());

    //Check if the first INPUT_SIZE elements are copied correctly
    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        assert(output[i] == 1.0f);
    }

    std::cout << "InputLayer test passed." << std::endl;
}

//Method to test the Hidden Layer and ensure proper data flow
void test_hidden_layer() {
    std::cout << "Testing HiddenLayer..." << std::endl;
    HiddenLayer hidden_layer(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE);

    std::vector<float> input(HIDDEN_LAYER1_SIZE, 1.0f);
    std::vector<float> output(OUTPUT_SIZE, 0.0f);

    hidden_layer.forward(input.data(), output.data());

    //Check if the output is non-zero (ReLU activation)
    for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
        assert(output[i] >= 0.0f);
    }

    std::cout << "HiddenLayer test passed." << std::endl;
}

//Method to test the Output Layer and ensure proper data flow
void test_output_layer() {
    std::cout << "Testing OutputLayer..." << std::endl;
    OutputLayer output_layer;

    std::vector<float> input(HIDDEN_LAYER1_SIZE, 1.0f);
    std::vector<float> output(OUTPUT_SIZE, 0.0f);

    output_layer.forward(input.data(), output.data());

    //Check if output is between 0 and 1 (sigmoid activation)
    for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
        assert(output[i] > 0.0f && output[i] < 1.0f);
    }

    std::cout << "OutputLayer test passed." << std::endl;
}

//Main Test Statement, feel free to adjust and add more
int main() {
    try {
        test_input_layer();
        test_hidden_layer();
        test_output_layer();

        std::cout << "All tests passed successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}