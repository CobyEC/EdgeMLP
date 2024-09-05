#pragma once

// utilities_Inference_Testbench.h
// Author: Coby Cockrell
// Date: 8/26/2024
// Purpose: This file declares utility functions for reading data and loading weights for inference.

#ifndef UTILITIES_INFERENCE_H
#define UTILITIES_INFERENCE_H

#include <vector>
#include <string>

// Function to read float data from a text file
std::vector<std::pair<std::vector<float>, int>> read_float_data(const std::string& file_path);

// Function to load weights from a file
std::vector<float> load_weights(const std::string& file_path);

#endif 