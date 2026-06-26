#include <inputs/input.hpp>

class PalindromeInput : public Input {
  public:
    PalindromeInput() : Input(4, 1, 20) {}
    std::vector<std::pair<std::vector<double>, unsigned int>> dataInput = {
        {{0, 0, 0, 0}, 1}, {{0, 0, 0, 1}, 0}, {{1, 0, 0, 1}, 1},
        {{0, 0, 1, 1}, 0}, {{1, 1, 1, 1}, 1}, {{0, 1, 1, 1}, 0},
        {{0, 1, 1, 0}, 1}, {{0, 0, 1, 0}, 1}, {{0, 1, 1, 0}, 1},
        {{1, 1, 1, 0}, 0}, {{1, 1, 1, 1}, 1}, {{1, 1, 0, 0}, 0},
        {{0, 0, 0, 0}, 1}, {{0, 1, 0, 0}, 0}, {{1, 0, 0, 1}, 1},
        {{1, 0, 0, 0}, 0}, {{1, 0, 0, 1}, 1}, {{1, 0, 1, 0}, 0},
        {{0, 0, 0, 0}, 1}, {{0, 1, 0, 1}, 0},
    };

    Mat inputSetup(size_t neuronIndex, size_t dataIndex) override;
    Mat outputSetup(size_t neuronIndex, size_t dataIndex) override;

    std::string getInputString(size_t dataIndex) override;
    std::string getOutputString(size_t dataIndex) override;
};
