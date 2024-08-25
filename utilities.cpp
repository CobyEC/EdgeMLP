// utils.cpp
// Author: Coby Cockrell
// Date: 5/1/2024
// Purpose: This file contains the implementation of utility functions for reading data, initializing weights, and saving/loading weights.

#include "utilities.h"

#include <fstream>
#include <sstream>
#include <random>
#include <ctime>


std::vector<std::pair<std::vector<int>, int>> read_data(const std::string& file_path) {
    std::vector<std::pair<std::vector<int>, int>> data;
    std::ifstream file(file_path);
    //Open the file
    if (file.is_open()) {
        std::string line;
        //Read each line
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<int> input;
            int label;
            std::string token;
            std::getline(ss, token, ',');
            for (char c : token) {
                input.push_back(c - '0');
            }
            ss >> label;
            data.emplace_back(input, label);
        }
        file.close();
    }
    return data;
}

void initialize_weights(std::vector<float>& weights, int num_weights, std::vector<float>& biases, int num_biases) {
    static std::mt19937 rng(std::time(nullptr));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    weights.resize(num_weights);
    for (int i = 0; i < num_weights; ++i) {
        weights[i] = dist(rng);
    }

    biases.resize(num_biases);
    for (int i = 0; i < num_biases; ++i) {
        biases[i] = dist(rng);
    }
}

void save_weights(const std::vector<float>& weights, const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (float weight : weights) {
            file << weight << "\n";
            //std::printf("Weight init...\n");
        }
        file.close();
    }
}

std::vector<float> load_weights(const std::string& file_path) {
    std::vector<float> weights;
    std::ifstream file(file_path);
    if (file.is_open()) {
        float weight;
        while (file >> weight) {
            weights.push_back(weight);
        }
        file.close();
    }
    return weights;
}