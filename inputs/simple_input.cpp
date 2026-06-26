#include <inputs/simple_input.hpp>

Mat SimpleInput::inputSetup(size_t neuronIndex, size_t dataIndex) {
    size_t index = indexes[dataIndex % 10];
    Mat d(1, 1);
    if (neuronIndex == dataInput.at(index)) {
        d(0, 0) = 1;
    } else {
        d(0, 0) = 0;
    }
    return d;
};
Mat SimpleInput::outputSetup(size_t neuronIndex, size_t dataIndex) {
    size_t index = indexes[dataIndex % 10];
    Mat d(1, 1);
    d(0, 0) = dataOutput.at(index);
    return d;
};

std::string SimpleInput::getInputString(size_t dataIndex) {
    size_t index = indexes[dataIndex % 10];
    return std::to_string(dataInput.at(index));
}
std::string SimpleInput::getOutputString(size_t dataIndex) {
    size_t index = indexes[dataIndex % 10];
    return std::to_string(dataOutput.at(index));
}
