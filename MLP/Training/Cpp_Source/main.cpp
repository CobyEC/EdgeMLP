// main.cpp
// Author: Coby Cockrell
// Date: 8/21/2024
// Purpose: This is main is to orchistrate and performing traing for the MLP class. 


#include "MLP.h"
#include "utilities.h"
//#include "test.txt"
//#include "train.txt"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <numeric>

//Method so I know inputs are properly converted toa float value
std::vector<float> convert_to_float(const std::vector<int>& input) {
    std::vector<float> float_input(input.begin(), input.end());
    std::cout << "Converted input: ";
    for (float val : float_input) std::cout << std::fixed << std::setprecision(4) << val << " ";
    std::cout << std::endl;
    return float_input;
}
//Method to average...
float average(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
}

//Main Method
int main() {
    try {
        std::cout << "Starting MLP training and testing..." << std::endl;

        //Initialize MLP with 9 input neurons
        MLP mlp(9);
        std::cout << "MLP initialized with 9 input neurons." << std::endl;

        std::vector<std::pair<std::vector<int>, int>> training_data = read_data("train.txt");
        std::cout << "Loaded " << training_data.size() << " training samples." << std::endl;

        if (training_data.empty()) {
            throw std::runtime_error("No training data loaded. Check your data file and loading process.");
        }

        //Print first few training samples
        std::cout << "First 5 training samples:" << std::endl;
        for (int i = 0; i < std::min(5, static_cast<int>(training_data.size())); ++i) {
            std::cout << "Sample " << i << ": Input = ";
            for (int val : training_data[i].first) {
                std::cout << val << " ";
            }
            std::cout << "Target = " << training_data[i].second << std::endl;
        }

        //Current best testing is 10 epochs with a lr = 0.001
        float learning_rate = 0.0001f;
        int epochs = 40;
        std::cout << "Learning rate: " << learning_rate << ", Epochs: " << epochs << std::endl;

        if (epochs <= 0) throw std::invalid_argument("Number of epochs must be positive");
        if (learning_rate <= 0.0f || learning_rate >= 1.0f) throw std::invalid_argument("Learning rate must be between 0 and 1");

        std::vector<float> errors(3);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\n--- Epoch " << epoch + 1 << "/" << epochs << " ---" << std::endl;
            float total_loss = 0.0f;
            int correct_predictions = 0;
            std::vector<float> hidden_activations;

            for (size_t i = 0; i < training_data.size(); ++i) {
                const auto& data_pair = training_data[i];
                const std::vector<int>& input = data_pair.first;
                int target = data_pair.second;

                std::vector<float> normalized_input = mlp.normalize_input(input);
                //if (i % 100 == 0) {  // Print every 100th sample
                    //std::cout << "Sample " << i << ": Normalized input: ";
                    //for (float val : normalized_input) std::cout << std::fixed << std::setprecision(4) << val << " ";
                    //std::cout << "Target: " << target << std::endl;
                //}

                mlp.forward(normalized_input);

                float output = mlp.get_output()[0];
                float loss = 0.5f * std::pow(output - target, 2);
                total_loss += loss;

                //Print every 100th sample
                if (i % 100 == 0) {  
                    std::cout << "Output: " << std::fixed << std::setprecision(6) << output << ", Loss: " << loss << std::endl;
                }

                if ((output > 0.5f && target == 1) || (output <= 0.5f && target == 0)) {
                    ++correct_predictions;
                }

                float error = output - target;

                errors[0] = clip_gradient(error, 1.0f);
                errors[1] = errors[0] * mlp.get_output_layer().get_output_derivative();
                errors[2] = errors[1] * mlp.get_hidden_layer1().get_output_derivative();

                //Print every 100th sample
                if (i % 100 == 0) {  
                    std::cout << "Errors: " << errors[0] << ", " << errors[1] << ", " << errors[2] << std::endl;
                }

                //Update Weights
                mlp.get_output_layer().update_weights(errors[0], learning_rate);
                mlp.get_hidden_layer1().update_weights(errors[1], learning_rate);
                mlp.get_input_layer().update_weights(errors[2], learning_rate);

                hidden_activations.push_back(mlp.get_hidden_layer1().get_output()[0]);
            }

            float accuracy = static_cast<float>(correct_predictions) / training_data.size();
            float avg_hidden_activation = average(hidden_activations);
            std::cout << "\nEpoch summary:" << std::endl;
            std::cout << "Total Loss: " << total_loss
                << ", Avg Loss: " << total_loss / training_data.size()
                << ", Accuracy: " << accuracy * 100 << "%"
                << ", Avg Hidden Activation: " << avg_hidden_activation << std::endl;

            //EarlyStopping
            if (accuracy > 0.96) {
                std::cout << "Early stopping: Achieved 96% accuracy." << std::endl;
                break;
            }
        }

        std::cout << "\nTraining completed." << std::endl;

        save_weights(mlp.get_weights(), "weights.txt");
        save_weights(mlp.get_biases(), "biases.txt");
        std::cout << "Weights and biases saved." << std::endl;

        std::vector<std::pair<std::vector<int>, int>> test_data = read_data("test.txt");
        std::cout << "\nTesting MLP with " << test_data.size() << " samples." << std::endl;
        int correct_predictions = 0;

        //Testing Loop
        for (size_t i = 0; i < test_data.size(); ++i) {
            const auto& data_pair = test_data[i];
            const std::vector<int>& input = data_pair.first;
            int target = data_pair.second;

            std::vector<float> float_input = convert_to_float(input);

            float prediction = mlp.predict(float_input);
            bool predicted_class = prediction > 0.5f;

            //Print every 10th test sample
            if (i % 10 == 0) {  
                std::cout << "Test Sample " << i << ": Prediction: " << prediction
                    << ", Predicted Class: " << predicted_class
                    << ", Actual Class: " << target << std::endl;
            }

            if ((predicted_class && target == 1) || (!predicted_class && target == 0)) {
                ++correct_predictions;
                if (i % 10 == 0) std::cout << "Correct prediction" << std::endl;
            }
            else {
                if (i % 10 == 0) std::cout << "Incorrect prediction" << std::endl;
            }
        }

        float test_accuracy = static_cast<float>(correct_predictions) / test_data.size() * 100;
        std::cout << "\nTest Results:" << std::endl;
        std::cout << "Correct Predictions: " << correct_predictions << "/" << test_data.size() << std::endl;
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nProgram completed successfully." << std::endl;
    return 0;
}