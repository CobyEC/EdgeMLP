#pragma once
//MLP.h
//Author: Coby Cockrell
//Date: 7/21/2024

#include "layers.h"
#include <vector>
#include <string>

class MLP {
public:
    MLP(uint32_t input_size);
    ~MLP();

    void forward(const std::vector<int>& input);
    float predict(const std::vector<int>& input);
    void train(const std::vector<std::pair<std::vector<int>, int>>& training_data, int epochs, float learning_rate);
    

private:
    InputLayer input_layer;
    HiddenLayer hidden_layer1;
    //HiddenLayer hidden_layer2;  //Optional Second hidden Layer
    OutputLayer output_layer;
    std::vector<float> intermediate;
    std::vector<float> intermediate1;
    std::vector<float> intermediate2;
    std::vector<float> output;
};