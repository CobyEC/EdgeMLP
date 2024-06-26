//utils.h
// Author: Coby Cockrell
// Date: 5/1/2024
// Purpose: This file is the header file for the utils group, as such the group facititates initial weight, reading, and writing operations

#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

//Function to read data from a text file
std::vector<std::pair<std::vector<int>, int>> read_data(const std::string& file_path);

//Function to initialize weights randomly
void initialize_weights(std::vector<float>& weights, int num_weights, std::vector<float>& biases, int num_biases);

//Function to save weights to a file
void save_weights(const std::vector<float>& weights, const std::string& file_path);

//Function to load weights from a file
std::vector<float> load_weights(const std::string& file_path);

#endif // UTILS_H