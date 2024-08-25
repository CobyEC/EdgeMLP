// MLP_Testbench.cpp
// Author: Coby Cockrell
// Date: 7/28/2024
// Purpose: This testbench shall test all individual components within the MLP.cpp file and overall this is the proto-main 

#include "mlp.h"
#include "utilities.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

//Generate random numbers for testing
std::vector<std::pair<std::vector<int>, int>> generate_random_data(int num_samples) {
    std::vector<std::pair<std::vector<int>, int>> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);

    for (int i = 0; i < num_samples; ++i) {
        int number = dis(gen);
        std::vector<int> digits;
        for (int j = 0; j < 6; ++j) {
            digits.push_back(number % 10);
            number /= 10;
        }
        std::reverse(digits.begin(), digits.end());

        //Simple primality test (not efficient, but okay for this example)
        bool is_prime = true;
        if (number <= 1) is_prime = false;
        for (int j = 2; j * j <= number; ++j) {
            if (number % j == 0) {
                is_prime = false;
                break;
            }
        }

        data.emplace_back(digits, is_prime ? 1 : 0);
    }
    return data;
}

//Test just the forward passes
void test_forward_pass(MLP& mlp) {
    try {
        std::cout << "Testing forward pass..." << std::endl;
        std::vector<int> input = { 1, 2, 3, 4, 5, 6 };
        std::cout << "Input: ";
        for (int i : input) std::cout << i << " ";
        std::cout << std::endl;
        float prediction = mlp.predict(input);
        std::cout << "Prediction for input [1,2,3,4,5,6]: " << prediction << std::endl;
        std::cout << "Forward pass test complete." << std::endl << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in forward pass test: " << e.what() << std::endl;
    }
}

//Test our train method
void test_training(MLP& mlp) {
    try {
        std::cout << "Testing training..." << std::endl;
        auto training_data = generate_random_data(100);
        mlp.train(training_data, 10, 0.01f); //10 epochs, learning rate 0.01
    }
    catch (const std::exception& e) {
        std::cerr << "Error in training test: " << e.what() << std::endl;
    }
}

//Test With a File Input
void test_with_file(MLP& mlp, const std::string& filename) {
    std::cout << "Testing with file: " << filename << std::endl;
    auto test_data = read_data(filename);
    int correct = 0;
    for (const auto& data_pair : test_data) {
        const std::vector<int>& input = data_pair.first;
        int target = data_pair.second;
        float prediction = mlp.predict(input);
        bool predicted_class = prediction > 0.5f;
        if (predicted_class == static_cast<bool>(target)) {  // Fixed comparison?
            correct++;
        }
        std::cout << "Input: ";
        for (int digit : input) std::cout << digit;
        std::cout << ", Target: " << target << ", Prediction: " << prediction
            << ", Predicted Class: " << predicted_class << std::endl;
    }
    float accuracy = static_cast<float>(correct) / test_data.size() * 100.0f;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
}

int main() {
    try {
        MLP mlp(6);

        // Test 1: Forward Pass
        test_forward_pass(mlp);
        std::cout << "\n\n\n-----------------------------------------------------------------------------------------\n" << "   ---  Finished Testing Forward Pass  ---   \n" << "\n-----------------------------------------------------------------------------------------\n\n\n";
        // Test 2: Training
        test_training(mlp);
        std::cout << "\n\n\n-----------------------------------------------------------------------------------------\n" << "   ---  Finished Testing Training  ---   \n" << "\n-----------------------------------------------------------------------------------------\n\n\n";
        // Test 3: Forward Pass after Training
        test_forward_pass(mlp);
        std::cout << "\n\n\n-----------------------------------------------------------------------------------------\n" << "   ---  Finished Testing Forward Training + Pass  ---   \n" << "\n-----------------------------------------------------------------------------------------\n\n\n";
        // Test 4: Test with file
        test_with_file(mlp, "test.txt");
        std::cout << "\n\n\n-----------------------------------------------------------------------------------------\n" << "   ---  Finished Testing File Interaction  ---   \n" << "\n-----------------------------------------------------------------------------------------\n\n\n";
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred in the main function: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}