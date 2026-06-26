#pragma once
#include <Eigen/Dense>
#include <cstddef>
typedef Eigen::MatrixXd Mat;

class Input {
  public:
    Input(size_t inputSize, size_t outputSize, size_t dataCount)
        : inputSize(inputSize), outputSize(outputSize), dataCount(dataCount) {}
    virtual Mat inputSetup(size_t, size_t) = 0;
    virtual Mat outputSetup(size_t, size_t) = 0;
    size_t inputSize;
    size_t outputSize;
    size_t dataCount;

    virtual std::string getInputString(size_t dataIndex) = 0;
    virtual std::string getOutputString(size_t dataIndex) = 0;
};
