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
#include <iomanip>
#include <numeric>

//Generate random float data for testing
std::vector<std::pair<std::vector<float>, int>> generate_random_data(int num_samples) {
    std::vector<std::pair<std::vector<float>, int>> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        std::vector<float> input;
        for (int j = 0; j < 9; ++j) {  //Generate 9 float inputs
            input.push_back(dis(gen));
        }

        //Simple classification based on sum of inputs
        float sum = std::accumulate(input.begin(), input.end(), 0.0f);
        int label = (sum > 4.5f) ? 1 : 0;  

        data.emplace_back(input, label);
    }
    return data;
}

//Test just the forward passes
void test_forward_pass(MLP& mlp) {
    try {
        std::cout << "Testing forward pass..." << std::endl;
        std::vector<float> input = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
        std::cout << "Input: ";
        for (float f : input) std::cout << std::fixed << std::setprecision(2) << f << " ";
        std::cout << std::endl;
        float prediction = mlp.predict(input);
        std::cout << "Prediction: " << std::fixed << std::setprecision(6) << prediction << std::endl;
        std::cout << "Forward pass test complete." << std::endl << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in forward pass test: " << e.what() << std::endl;
    }
}

//Test With a File Input
void test_with_file(MLP& mlp, const std::string& filename) {
    std::cout << "Testing with file: " << filename << std::endl;
    auto test_data = read_float_data(filename);
    int correct = 0;
    for (const auto& data_pair : test_data) {
        const std::vector<float>& input = data_pair.first;
        int target = data_pair.second;
        float prediction = mlp.predict(input);
        bool predicted_class = prediction > 0.5f;
        if (predicted_class == static_cast<bool>(target)) {
            correct++;
        }
        std::cout << "Input: ";
        for (float val : input) std::cout << std::fixed << std::setprecision(2) << val << " ";
        std::cout << ", Target: " << target << ", Prediction: " << std::fixed << std::setprecision(6) << prediction
            << ", Predicted Class: " << predicted_class << std::endl;
    }
    float accuracy = static_cast<float>(correct) / test_data.size() * 100.0f;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
}

int main() {
    try {
        MLP mlp(9);  //Initialize with 9 input neurons

        //Test 1: Forward Pass
        test_forward_pass(mlp);
        std::cout << "\n-----------------------------------------------------------------------------------------\n"
            << "   ---  Finished Testing Forward Pass  ---   \n"
            << "-----------------------------------------------------------------------------------------\n\n";

        //Test 2: Test with file
        test_with_file(mlp, "test.txt");
        std::cout << "\n-----------------------------------------------------------------------------------------\n"
            << "   ---  Finished Testing File Interaction  ---   \n"
            << "-----------------------------------------------------------------------------------------\n\n";

        //Additional test: Random data
        auto random_data = generate_random_data(100);
        int correct = 0;
        for (const auto& data_pair : random_data) {
            float prediction = mlp.predict(data_pair.first);
            bool predicted_class = prediction > 0.5f;
            if (predicted_class == static_cast<bool>(data_pair.second)) {
                correct++;
            }
        }
        float accuracy = static_cast<float>(correct) / random_data.size() * 100.0f;
        std::cout << "Random Data Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred in the main function: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}