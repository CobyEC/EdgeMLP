// act_Testbench.cpp
// Author: Coby Cockrell
// Date: 5/8/2024
// Purpose: Testbench for activation functions

#include <iostream>
#include <cmath>
#include <vector>
#include "activate.h"

// Helper function to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) < epsilon;
}

// Test ReLU function
void test_relu() {
    std::cout << "Testing ReLU function..." << std::endl;
    std::vector<float> inputs = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
    std::vector<float> expected = { 0.0f, 0.0f, 0.0f, 1.0f, 2.0f };

    for (size_t i = 0; i < inputs.size(); ++i) {
        float result = activate::relu(inputs[i]);
        if (approx_equal(result, expected[i])) {
            std::cout << "PASS: ReLU(" << inputs[i] << ") = " << result << std::endl;
        }
        else {
            std::cout << "FAIL: ReLU(" << inputs[i] << ") = " << result << ", expected " << expected[i] << std::endl;
        }
    }
}

// Test Sigmoid function
void test_sigmoid() {
    std::cout << "Testing Sigmoid function..." << std::endl;
    std::vector<float> inputs = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
    std::vector<float> expected = { 0.119203f, 0.268941f, 0.5f, 0.731059f, 0.880797f };

    for (size_t i = 0; i < inputs.size(); ++i) {
        float result = activate::sigmoid(inputs[i]);
        if (approx_equal(result, expected[i])) {
            std::cout << "PASS: Sigmoid(" << inputs[i] << ") = " << result << std::endl;
        }
        else {
            std::cout << "FAIL: Sigmoid(" << inputs[i] << ") = " << result << ", expected " << expected[i] << std::endl;
        }
    }
}

// Test Softmax function
void test_softmax() {
    std::cout << "Testing Softmax function..." << std::endl;
    std::vector<float> input = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expected = { 0.0320586f, 0.0871443f, 0.236883f, 0.644014f };
    std::vector<float> output(input.size());

    activate::softmax(input.data(), output.data(), input.size());

    bool all_pass = true;
    for (size_t i = 0; i < input.size(); ++i) {
        if (approx_equal(output[i], expected[i])) {
            std::cout << "PASS: Softmax[" << i << "] = " << output[i] << std::endl;
        }
        else {
            std::cout << "FAIL: Softmax[" << i << "] = " << output[i] << ", expected " << expected[i] << std::endl;
            all_pass = false;
        }
    }

    if (all_pass) {
        std::cout << "Softmax test passed for all elements." << std::endl;
    }
}

int main() {
    test_relu();
    std::cout << std::endl;
    test_sigmoid();
    std::cout << std::endl;
    test_softmax();

    return 0;
}