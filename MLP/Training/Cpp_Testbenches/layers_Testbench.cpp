//layers_Testbench.cpp
//Author: Coby Cockrell
//Date:5/9/2024
//Purpose: This file is the current testbench for the layer.cpp/layer.h files, I simply verify they take input and produce an expected output.


#include "layers.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

//Method to test the Input Layer and ensure proper data flow
void test_input_layer() {
    std::cout << "Testing InputLayer..." << std::endl;
    InputLayer input_layer(INPUT_SIZE, HIDDEN_LAYER1_SIZE);

    std::vector<float> input(INPUT_SIZE, 1.0f);
    std::vector<float> output(HIDDEN_LAYER1_SIZE);

    input_layer.forward(input.data(), output.data());

    //Check if the first INPUT_SIZE elements are copied correctly
    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        assert(output[i] == 1.0f);
    }

    //Check if the remaining elements are zero
    for (uint32_t i = INPUT_SIZE; i < HIDDEN_LAYER1_SIZE; ++i) {
        assert(output[i] == 0.0f);
    }

    std::cout << "InputLayer test passed." << std::endl;
}

//Method to test the Hidden Layer and ensure proper data flow
void test_hidden_layer() {
    std::cout << "Testing HiddenLayer..." << std::endl;
    try {
        HiddenLayer hidden_layer(HIDDEN_LAYER1_SIZE, OUTPUT_SIZE);

        std::cout << "HiddenLayer created with input size: " << HIDDEN_LAYER1_SIZE
            << ", output size: " << OUTPUT_SIZE << std::endl;

        std::cout << "Weights size: " << hidden_layer.get_weights().size() << std::endl;
        std::cout << "Biases size: " << hidden_layer.get_biases().size() << std::endl;

        //Print some weights and biases
        std::cout << "Sample weights: ";
        for (int i = 0; i < std::min(5, (int)hidden_layer.get_weights().size()); ++i) {
            std::cout << hidden_layer.get_weights()[i] << " ";
        }
        std::cout << "\nSample biases: ";
        for (int i = 0; i < std::min(5, (int)hidden_layer.get_biases().size()); ++i) {
            std::cout << hidden_layer.get_biases()[i] << " ";
        }
        std::cout << std::endl;

        std::vector<float> input(HIDDEN_LAYER1_SIZE, 1.0f);
        std::vector<float> output(OUTPUT_SIZE);

        std::cout << "Input size: " << input.size() << ", Output size: " << output.size() << std::endl;

        hidden_layer.forward(input.data(), output.data());

        std::cout << "Forward pass completed." << std::endl;

        //Print all output values
        std::cout << "Output values: ";
        for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "HiddenLayer test passed." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in HiddenLayer test: " << e.what() << std::endl;
    }
}

//Method to test the Output Layer and ensure proper data flow
void test_output_layer() {
    std::cout << "Testing OutputLayer..." << std::endl;
    OutputLayer output_layer;

    std::vector<float> input(HIDDEN_LAYER1_SIZE, 1.0f);
    std::vector<float> output(OUTPUT_SIZE);

    output_layer.forward(input.data(), output.data());

    //Check if output is between 0 and 1 (sigmoid activation)
    for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
        assert(output[i] > 0.0f && output[i] < 1.0f);
    }

    //Test weight update
    float error = 0.1f;
    float learning_rate = 0.01f;
    output_layer.update_weights(error, learning_rate);

    //Test output derivative
    float derivative = output_layer.get_output_derivative();
    assert(derivative > 0.0f && derivative < 0.25f);  //Sigmoid derivative is always in this range

    std::cout << "OutputLayer test passed." << std::endl;
}

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
