#pragma once
// MLP.h
// Author: Coby Cockrell
// Date: 7/21/2024
// Purpose: This header file declares functions relates to the MLP such as layers, get/put methods, and internal variables

#include "layers.h"
#include <vector>
#include <string>

class MLP {
public:
    MLP(uint32_t input_size);
    ~MLP();

    void forward(const std::vector<float>& input) const;
    float predict(const std::vector<float>& input) const;

    std::vector<float> get_weights() const;
    std::vector<float> get_biases() const;
    const std::vector<float>& get_output() const;
    const Layer& get_input_layer() const;
    const Layer& get_hidden_layer1() const;
    const Layer& get_output_layer() const;
    Layer& get_input_layer();
    Layer& get_hidden_layer1();
    Layer& get_output_layer();

    static std::vector<float> normalize_input(const std::vector<int>& input);

private:
    InputLayer input_layer;
    HiddenLayer hidden_layer1;
    OutputLayer output_layer;
    mutable std::vector<float> intermediate;
    mutable std::vector<float> output;
};