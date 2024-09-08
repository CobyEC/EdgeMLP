// main.cpp
// Author: Coby Cockrell
// Date: 8/21/2024
// Purpose: This is main is to orchistrate and performing traing for the MLP class. 
#include "MLP.h"
#include "utilities.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

// Function to read data from a file and return a vector of pairs
std::vector<std::pair<std::vector<float>, int>> read_data_from_file(const std::string& file_path) {
    std::vector<std::pair<std::vector<float>, int>> data;
    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int number, target;
        char delimiter;
        if (ss >> number >> delimiter >> target) {
            // Create a feature vector with the number and its divisibility flag
            std::vector<float> features;
            features.push_back(static_cast<float>(number)); // Original number
            features.push_back(number % 2 == 0 ? 1.0f : 0.0f); // Divisibility flag

            data.emplace_back(features, target);
        }
        else {
            std::cerr << "Warning: Invalid line format: " << line << std::endl;
        }
    }

    if (data.empty()) {
        std::cerr << "Warning: No data read from file. Check file format and content." << std::endl;
    }

    file.close();
    return data;
}

// Function to evaluate the model on a dataset
void evaluate_model(MLP& mlp, const std::vector<std::pair<std::vector<float>, int>>& data) {
    int correct_predictions = 0;
    for (const auto& sample : data) {
        const std::vector<float>& input_features = sample.first;
        int target = sample.second;

        mlp.forward(input_features);

        float output = mlp.get_output()[0];
        bool predicted_class = output > 0.5f;
        if (predicted_class == target) {
            ++correct_predictions;
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / data.size();
    std::cout << "Validation Accuracy: " << accuracy * 100 << "%" << std::endl;
}

// Main Method
int main() {
    try {
        std::cout << "Starting MLP training and testing from file..." << std::endl;

        // Initialize MLP with 2 input neurons (number and divisibility flag)
        MLP mlp(2);
        std::cout << "MLP initialized with 2 input neurons." << std::endl;

        // Read training data from file
        std::vector<std::pair<std::vector<float>, int>> training_data = read_data_from_file("train.txt");
        std::cout << "Loaded " << training_data.size() << " training samples from file." << std::endl;

        float learning_rate = 0.1f; // Adjust learning rate
        int epochs = 20;
        std::cout << "Learning rate: " << learning_rate << ", Epochs: " << epochs << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\n--- Epoch " << epoch + 1 << "/" << epochs << " ---" << std::endl;
            float total_loss = 0.0f;
            int correct_predictions = 0;

            for (const auto& sample : training_data) {
                const std::vector<float>& input_features = sample.first;
                int target = sample.second;

                // Normalize inputs if necessary
                std::vector<float> normalized_input = input_features; // Apply normalization if needed

                mlp.forward(normalized_input);

                float output = mlp.get_output()[0];
                float loss = 0.5f * std::pow(output - target, 2);
                total_loss += loss;

                // Determine if prediction is correct
                bool predicted_class = output > 0.5f;
                if (predicted_class == target) {
                    ++correct_predictions;
                }

                // Calculate error and update weights
                float error = output - target;
                std::vector<float> errors(1, error);

                mlp.get_output_layer().update_weights(errors[0], learning_rate);
                mlp.get_hidden_layer1().update_weights(errors[0], learning_rate);
                mlp.get_input_layer().update_weights(errors[0], learning_rate);

                // Debug prints
                std::cout << "Input: " << input_features[0] << ", Target: " << target
                    << ", Output: " << output << ", Error: " << error << std::endl;
            }

            float accuracy = static_cast<float>(correct_predictions) / training_data.size();
            std::cout << "Total Loss: " << total_loss
                << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
        }

        std::cout << "\nTraining completed." << std::endl;

        // Read test data from file and evaluate
        std::vector<std::pair<std::vector<float>, int>> test_data = read_data_from_file("test.txt");
        std::cout << "Loaded " << test_data.size() << " test samples from file." << std::endl;
        evaluate_model(mlp, test_data);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nProgram completed successfully." << std::endl;
    return 0;
}
