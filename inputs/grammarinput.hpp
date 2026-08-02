#include <inputs/input.hpp>
#include <vector>

const double PI = 3.141592653589793;
class GrammarInput : public Input {
  public:
    GrammarInput() : Input(1, 1, 5) {}
    std::vector<std::vector<std::vector<double>>> dataInput = {
        {{0, 1, 1}, {0, 0, 1}},
        {
            {0, 0, 0, 1},
            {0, -1, -1, -1},
        },
        {
            {0, 1, 0, 1, 1},
            {0, 0, 0, 0, 1},
        },
        {
            {0, 1, 1, 1},
            {0, 0, 1, -1},
        },
        {{0, 1, 1, 0, 0, 1}, {0, 0, 1, -1, -1, -1}}};
    Mat inputSetup(size_t neuronIndex, size_t dataIndex) override;
    Mat outputSetup(size_t neuronIndex, size_t dataIndex) override;

    std::string getInputString(size_t dataIndex) override;
    std::string getOutputString(size_t dataIndex) override;
};
