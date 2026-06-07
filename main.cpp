#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <imgui-SFML.h>
#include <imgui.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#ifdef _WIN32
#include <windows.h>
#endif

typedef Eigen::MatrixXd Matrix;

class Neuron;

struct Convert {
    Matrix outer;
    Matrix inner;
};
struct ProtoConnection {
    Matrix activityWeights;
    Matrix predictionWeights;
    Convert convertActivity;
};

struct Connection {
    ProtoConnection lower;
    ProtoConnection higher;
    ProtoConnection actual;
    double value;
};

struct OutgoingConnection {
    Connection *incomingConnection;
};

static Matrix convertInput(const Matrix &m, const Convert &convert) {
    // inner rows match input m cols and inner cols match resulting m cols
    // outer rows match resulting m rows and outer cols match input m rows
    return convert.outer * (m * convert.inner);
}
double totalError = 0;
const double connectionValueLimit = 2;
class Neuron {
  protected:
    std::map<Neuron *, Connection *> possibleConnections;
    Matrix activity;
    Matrix prediction;
    Matrix error;

  public:
    bool isInput;
    std::map<Neuron *, Connection> incoming = {};
    std::map<Neuron *, OutgoingConnection> outgoing = {};
    std::string name;

    Neuron(size_t rows, size_t cols, bool isInput = false) {
        activity = Matrix::Zero(rows, cols);
        prediction = Matrix::Zero(rows, cols);
        error = Matrix::Zero(rows, cols);
        this->isInput = isInput;
    }

    Matrix getActivity() { return this->activity; }

    void setValues(Matrix activity) {
        const Matrix inner =
            Matrix::Ones(activity.outerSize(), this->activity.outerSize());
        const Matrix outer =
            Matrix::Ones(this->activity.innerSize(), activity.innerSize());
        const auto converted = convertInput(activity, {outer, inner});
        this->prediction = converted;
        this->activity = converted;
    }

    void recalcWeights() {
        for (auto &[neuron, connection] : incoming) {
            connection.actual.activityWeights +=
                (this->error * neuron->activity) * 0.1;
        }

        // C2*(IAi*C1) = ACi
        // P = sum(AC*Wi)
        // E = TAj - sum((C2 x (IAi x C1)) * Wi)
    }

    void addPossibleConnection(Neuron *neuron) {
        const auto &rows = neuron->activity.innerSize();
        const auto &cols = neuron->activity.outerSize();
        const auto higher = ProtoConnection{
            .activityWeights = Matrix::Ones(rows, cols),
            .predictionWeights = Matrix::Ones(rows, cols),
            .convertActivity =
                {
                    .outer = Matrix::Ones(this->activity.innerSize(), rows),
                    .inner = Matrix::Ones(cols, this->activity.outerSize()),
                },
        };
        const auto lower = ProtoConnection{
            .activityWeights = Matrix::Ones(rows, cols),
            .predictionWeights = Matrix::Ones(rows, cols),
            .convertActivity =
                {
                    .outer = Matrix::Ones(this->activity.innerSize(), rows),
                    .inner = Matrix::Ones(cols, this->activity.outerSize()),
                },
        };
        const auto newConnection = ProtoConnection{
            .activityWeights = Matrix::Ones(rows, cols),
            .predictionWeights = Matrix::Ones(rows, cols),
            .convertActivity =
                {
                    .outer = Matrix::Ones(this->activity.innerSize(), rows),
                    .inner = Matrix::Ones(cols, this->activity.outerSize()),
                },
        };
        Connection *newPossibleConnection = new Connection{
            .lower = lower,
            .higher = higher,
            .actual = newConnection,
            .value = connectionValueLimit,
        };
        // this -> neuron
        possibleConnections.insert({neuron, newPossibleConnection});
        // neuron -> this
        neuron->possibleConnections.insert({this, newPossibleConnection});
    }

    void recalcActivity() {
        if (!isInput) {
            recalcPrediction();
            recalcError();
            Matrix delta = -error;

            for (const auto &[neuron, connection] : outgoing) {
                const auto thisWeights =
                    neuron->incoming.at(this).actual.activityWeights;

                delta += thisWeights.cwiseProduct(neuron->error);
            }
            this->activity += delta * 0.1;
        }

        std::vector<Neuron *> toErase = {};
        for (auto &[neuron, posCon] : possibleConnections) {
            const auto activity =
                convertInput(this->activity, posCon->actual.convertActivity);

            const auto res = posCon->actual.predictionWeights.cwiseProduct(
                (neuron->activity -
                 (neuron->prediction +
                  activity.cwiseProduct(posCon->actual.activityWeights) *
                      posCon->value)));
            const double avg = res.sum() / (double)res.size();
            posCon->value += avg;
            posCon->value =
                posCon->value > connectionValueLimit    ? connectionValueLimit
                : posCon->value < -connectionValueLimit ? -connectionValueLimit
                                                        : posCon->value;
            // std::cout << "connection value " << this->name << " -> "
            //           << neuron->name << ": " << posCon->value << ", " << avg
            //           << "\n";
            if (posCon->value < 0.2 && posCon->value > -0.2) {
                neuron->incoming.insert({this, *posCon});
                outgoing.insert({neuron, OutgoingConnection{
                                             .incomingConnection =
                                                 &neuron->incoming.at(this)}});
                // std::free(posCon);
                toErase.push_back(neuron);
            }
        }
        for (auto neuron : toErase) {
            possibleConnections.erase(neuron);
            // neuron->possibleConnections.erase(this);

            // std::cout << "connection executed " << this->name << " -> "
            //           << neuron->name << "\n";
        }
    }

    void recalcPrediction() {
        this->prediction.setZero();
        for (const auto &[neuron, connection] : incoming) {
            const auto activity = convertInput(
                neuron->activity, connection.actual.convertActivity);
            this->prediction +=
                activity.cwiseProduct(connection.actual.predictionWeights);
        }
    }

    void recalcError() {
        if (isInput) {
            return;
        }
        totalError -= this->error.sum() / this->error.size();
        this->error = (activity - prediction);
        totalError += this->error.sum() / this->error.size();
    }
};

std::vector<double> dataInput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<double> dataOutput = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

int main() {
    sf::RenderWindow window(sf::VideoMode(1280, 720), "SFML + ImGui");
    window.setFramerateLimit(60);

    if (!ImGui::SFML::Init(window))
        return -1;

    sf::Clock deltaClock;

    std::vector<Neuron *> inputs = {new Neuron(1, 1, true)};
    std::vector<Neuron *> outputs = {new Neuron(1, 1, true)};

    std::vector<Neuron *> neurons = {};

    inputs.at(0)->name = "input";
    outputs.at(0)->name = "output";

    for (size_t i = 0; i < 4; i++) {
        neurons.push_back(new Neuron(1, 1));
        neurons.at(neurons.size() - 1)->name = "n" + std::to_string(i);
    }
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            if (i == j) {
                continue;
            }
            neurons.at(i)->addPossibleConnection(neurons.at(j));
        }
        for (size_t j = 0; j < inputs.size(); j++) {
            inputs.at(j)->addPossibleConnection(neurons.at(i));
            neurons.at(i)->addPossibleConnection(inputs.at(j));
        }
        for (size_t j = 0; j < outputs.size(); j++) {
            outputs.at(j)->addPossibleConnection(neurons.at(i));
            neurons.at(i)->addPossibleConnection(outputs.at(j));
        }
    }

    double lastError = 1;
    const size_t indexes[] = {7, 0, 5, 9, 3, 2, 4, 1, 6, 8};
    for (size_t i = 0; i < 1; i++) {
        for (size_t k = 0; k < 100; k++) {
            const size_t index = indexes[i];
            for (size_t j = 0; j < inputs.size(); j++) {
                std::set<Neuron *> visited = {};
                std::set<Neuron *> children = {};
                double input[] = {dataInput.at(index) / 10.0};
                double output[] = {dataOutput.at(index)};
                inputs.at(j)->setValues(Eigen::Map<Matrix>(input, 1, 1));
                outputs.at(j)->setValues(Eigen::Map<Matrix>(output, 1, 1));
                children.insert(inputs.at(j));
                // std::cout << "out" << inputs.at(0)->outgoing.size();
                while (children.size() > 0) {
                    auto child = *children.begin();
                    visited.insert(child);
                    child->recalcActivity();
                    for (auto &c : child->outgoing) {
                        if (visited.find(c.first) == visited.end()) {
                            children.insert(c.first);
                        }
                    }
                    children.erase(child);
                }
            }
            std::cout << "pass " << k << "===================\n"
                      << std::fixed << std::setprecision(2) << totalError
                      << totalError - lastError << "\n";
            if ((abs(totalError - lastError)) <= 0) {
                break;
            }
            lastError = totalError;
        }
    }

    {
        // for (size_t i = 0; i < 4; i++) {
        //     std::set<Neuron *> visited = {};
        //     std::set<Neuron *> children = {};
        //     for (size_t j = 0; j < inputs.size(); j++) {
        //         double data[] = {dataInput.at(0) / 10.0};
        //         inputs.at(j)->setValues(Eigen::Map<Matrix>(data, 1, 1));
        //         children.insert(inputs.at(j));
        //     }
        //     for (size_t j = 0; j < outputs.size(); j++) {
        //         outputs.at(j)->isInput = false;
        //     }

        //     while (children.size() > 0) {
        //         auto child = *children.begin();
        //         visited.insert(child);
        //         child->recalcActivity();
        //         for (auto &c : child->outgoing) {
        //             if (visited.find(c.first) == visited.end()) {
        //                 children.insert(c.first);
        //             }
        //         }
        //         children.erase(child);
        //     }
        // }
        // std::cout << "number of incoming: " << outputs.at(0)->incoming.size()
        //           << "\n";
        // std::cout << "number of incoming: " << inputs.at(0)->incoming.size()
        //           << "\n";
        // const double res = outputs.at(0)->getActivity()(0, 0);
        // std::cout << "result: " << res << ",  " << (res < 0.5 ? 0 : 1) <<
        // "\n";
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        // Your ImGui UI
        ImGui::Begin("Hello");
        // for (size_t i = 0; i < inputs.size(); i++) {

        // ImGui::Text(inputs.at(i)->name+(inputs.at()));
        // }
        ImGui::Text("SFML");
        ImGui::End();

        window.clear();
        // Draw your SFML stuff here
        ImGui::SFML::Render(window); // Render ImGui on top
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}