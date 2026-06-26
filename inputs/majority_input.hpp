#include <inputs/input.hpp>

class MajorityInput : public Input {
  public:
    MajorityInput() : Input(3, 1, 8) {}
    std::vector<std::vector<double>> dataInput = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1},
        {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}};
    std::vector<double> dataOutput = {0, 0, 1, 1, 0, 1, 0, 1};

    Mat inputSetup(size_t neuronIndex, size_t dataIndex) override;
    Mat outputSetup(size_t neuronIndex, size_t dataIndex) override;

    std::string getInputString(size_t dataIndex) override;

    std::string getOutputString(size_t dataIndex) override;
};
