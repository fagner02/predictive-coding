#include <inputs/palindrome_input.hpp>
#include <sstream>

Mat PalindromeInput::inputSetup(size_t neuronIndex, size_t dataIndex) {
    Mat d(1, 1);
    d(0, 0) = dataInput[dataIndex % dataCount].first[neuronIndex];
    return d;
};
Mat PalindromeInput::outputSetup(size_t neuronIndex, size_t dataIndex) {
    Mat d(1, 1);
    d(0, 0) = dataInput[dataIndex % dataCount].second;
    return d;
};

std::string PalindromeInput::getInputString(size_t dataIndex) {
    size_t index = dataIndex % dataCount;
    std::stringstream ss;
    ss << dataInput[index].first[0] << ", " << dataInput[index].first[1] << ", "
       << dataInput[index].first[2] << ", " << dataInput[index].first[3];
    return ss.str();
}
std::string PalindromeInput::getOutputString(size_t dataIndex) {
    size_t index = dataIndex % dataCount;
    return std::to_string(dataInput.at(index).second);
}
