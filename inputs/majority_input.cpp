#include <inputs/majority_input.hpp>

Mat MajorityInput::inputSetup(size_t neuronIndex, size_t dataIndex) {
    const size_t index = dataIndex % 8;

    Mat d(1, 1);
    d(0, 0) = dataInput[index][neuronIndex];
    return d;
}
Mat MajorityInput::outputSetup(size_t neuronIndex, size_t dataIndex) {
    const size_t index = dataIndex % 8;

    Mat d(1, 1);
    d(0, 0) = dataOutput.at(index);
    return d;
}

std::string MajorityInput::getInputString(size_t dataIndex) {
    std::stringstream ss;
    const size_t index = dataIndex % 8;
    ss << dataInput[index][0] << ", " << dataInput[index][1] << ", "
       << dataInput[index][2];
    return ss.str();
}

std::string MajorityInput::getOutputString(size_t dataIndex) {
    const size_t index = dataIndex % 8;
    return std::to_string(dataOutput[index]);
}
