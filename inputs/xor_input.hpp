#pragma once
#include <inputs/input.hpp>

class XorInput : public Input {
  public:
    XorInput() : Input(2, 1, 4) {}
    std::vector<double> dataInputA = {0, 0, 1, 1};
    std::vector<double> dataInputB = {0, 1, 0, 1};
    std::vector<double> dataOutput = {0, 1, 1, 0};
    size_t xorIndexes[12] = {1, 0, 3, 2, 0, 3, 1, 2, 3, 0, 2, 1};

    Mat inputSetup(size_t neuronIndex, size_t dataIndex) override;
    Mat outputSetup(size_t neuronIndex, size_t dataIndex) override;

    std::string getInputString(size_t dataIndex) override;

    std::string getOutputString(size_t dataIndex) override;
};
