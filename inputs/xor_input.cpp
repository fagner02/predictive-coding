#include <inputs/xor_input.hpp>

Mat XorInput::inputSetup(size_t neuronIndex, size_t dataIndex) {
    const size_t index = xorIndexes[dataIndex % 12];

    Mat d(1, 1);
    d(0, 0) = neuronIndex ? dataInputA[index] : dataInputB[index];
    return d;
}
Mat XorInput::outputSetup(size_t neuronIndex, size_t dataIndex) {
    const size_t index = xorIndexes[dataIndex % 12];

    Mat d(1, 1);
    d(0, 0) = dataOutput.at(index);
    return d;
}

std::string XorInput::getInputString(size_t dataIndex) {
    std::stringstream ss;
    const size_t index = xorIndexes[dataIndex % 12];
    ss << dataInputA[index] << ", " << dataInputB[index];
    return ss.str();
}

std::string XorInput::getOutputString(size_t dataIndex) {
    const size_t index = xorIndexes[dataIndex % 12];
    return std::to_string(dataOutput[index]);
}
