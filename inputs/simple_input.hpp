#include <inputs/input.hpp>

class SimpleInput : public Input {
  public:
    const size_t indexes[10] = {7, 0, 5, 9, 3, 2, 4, 1, 6, 8};
    SimpleInput() : Input(1, 1, 10) {}
    std::vector<double> dataInput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // std::vector<double> dataOutput = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    std::vector<double> dataOutput = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    // std::vector<double> dataOutput = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    // std::vector<double> dataOutput = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
    Mat inputSetup(size_t neuronIndex, size_t dataIndex) override;
    Mat outputSetup(size_t neuronIndex, size_t dataIndex) override;

    std::string getInputString(size_t dataIndex) override;
    std::string getOutputString(size_t dataIndex) override;
};
