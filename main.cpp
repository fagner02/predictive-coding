#include <SFML/Graphics.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <imgui-SFML.h>
#include <imgui.h>
#include <implot.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

typedef Eigen::MatrixXd Matrix;

class Neuron;

struct Convert {
    Matrix outer;
    Matrix inner;
};
struct ProtoConnection {
    Matrix activityWeights;
};

struct Connection {
    ProtoConnection actual;
    double value;
};

struct OutgoingConnection {
    Connection *incoming;
};

const double connectionValueLimit = 2;

struct RecalcResponse {
    Matrix oldError;
    Matrix newError;
};

class Neuron {
  protected:
    std::map<Neuron *, Connection *> possibleConnections;
    Matrix activity;
    Matrix prediction;
    Matrix error;
    Matrix sd;

  public:
    bool isInput;
    std::map<Neuron *, Connection *> incoming = {};
    std::map<Neuron *, OutgoingConnection> outgoing = {};
    std::string name;

    Neuron(size_t rows, size_t cols, bool isInput = false) {
        activity = Matrix::Zero(rows, cols);
        prediction = Matrix::Ones(rows, cols);
        error = Matrix::Ones(rows, cols);
        this->isInput = isInput;
    }

    Matrix getActivity() { return this->activity; }

    Matrix getPrediction() { return this->prediction; }

    void reset() {
        size_t rows = activity.rows();
        size_t cols = activity.cols();

        activity = Matrix::Zero(rows, cols);
        prediction = Matrix::Ones(rows, cols);
        error = Matrix::Ones(rows, cols);
    }

    void setValues(Matrix activity) {
        this->prediction = activity;
        this->activity = activity;
    }

    void recalcWeights() {
        for (auto &[neuron, connection] : incoming) {
            connection->actual.activityWeights +=
                ((this->error.cwiseProduct(
                      (this->sd).cwiseProduct(neuron->activity)) *
                  1));
            if (this->name == "output") {
                std::cout << std::fixed << std::setprecision(4)
                          << "target=" << this->activity
                          << " pred=" << this->prediction
                          << " err=" << this->error << "\n";
            }
        }
    }

    std::map<Neuron *, Connection *> getPossibleConnections() {
        return this->possibleConnections;
    }

    void addPossibleConnection(Neuron *neuron) {
        const auto &rows = neuron->activity.rows();
        const auto &cols = neuron->activity.cols();
        const double ratio =
            1.0 / (this->activity.rows() * this->activity.cols());
        const auto newConnection = ProtoConnection{
            .activityWeights = Matrix::Random(rows, cols) * 0.1,
        };
        Connection *newPossibleConnection = new Connection{
            .actual = newConnection,
            .value = 0,
        };
        // this -> neuron
        possibleConnections.insert({neuron, newPossibleConnection});
    }

    Matrix sigmoidDerivative(Matrix m) {
        auto e = (-m.array()).exp();
        auto res = (1 + e);
        return (e / (res * res)).matrix();
    }

    Matrix sigmoid(Matrix m) { return (1 / (1 + (-m.array()).exp())).matrix(); }

    void recalcConnectionValue(Neuron *neuron, Connection *connection) {
        Matrix ta = sigmoid(
            this->activity.cwiseProduct(connection->actual.activityWeights));

        const auto res =
            ta.cwiseProduct(neuron->error).cwiseProduct(neuron->prediction);
        const double avg = res.sum();
        connection->value += avg;
        connection->value =
            connection->value > connectionValueLimit    ? connectionValueLimit
            : connection->value < -connectionValueLimit ? -connectionValueLimit
                                                        : connection->value;
    }

    void recalcActivity() {
        if (!isInput) {

            Matrix delta = -error;

            for (const auto &[neuron, connection] : outgoing) {
                const auto thisWeights =
                    connection.incoming->actual.activityWeights;

                delta += thisWeights.cwiseProduct(
                    neuron->error.cwiseProduct(neuron->sd));

                recalcConnectionValue(neuron, connection.incoming);
            }
            this->activity += delta;
        }
    }
    RecalcResponse update() {
        recalcPrediction();
        const Matrix oldError = this->error;
        recalcError();
        const Matrix newError = this->error;
        recalcActivity();
        std::vector<Neuron *> toErase = {};
        for (auto &[neuron, posCon] : possibleConnections) {

            recalcConnectionValue(neuron, posCon);

            if (posCon->value > 0.0) {
                toErase.push_back(neuron);
                neuron->incoming.insert({this, new Connection(*posCon)});
                outgoing.insert(
                    {neuron, OutgoingConnection{
                                 .incoming = neuron->incoming.at(this)}});
            }
        }
        for (auto neuron : toErase) {
            possibleConnections.erase(neuron);
        }
        return {oldError, newError};
    }

    void recalcPrediction() {
        this->prediction.setZero();
        for (const auto &[neuron, connection] : incoming) {
            this->prediction += neuron->activity.cwiseProduct(
                connection->actual.activityWeights);
        }
        sd = sigmoidDerivative(this->prediction);
        this->prediction = sigmoid(this->prediction);
    }

    void recalcError() { this->error = (activity - prediction); }
};

std::vector<double> dataInput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<double> dataOutput = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
//  std::vector<double> dataOutput = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
//  std::vector<double> dataOutput = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
//  std::vector<double> dataOutput = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1};

const size_t indexes[] = {7, 0, 5, 9, 3, 2, 4, 1, 6, 8};
class Network {
  public:
    std::vector<Neuron *> inputs = {};
    std::vector<Neuron *> outputs = {new Neuron(1, 1, true)};
    std::vector<Neuron *> neurons = {};
    std::vector<Neuron *> biases = {};
    std::vector<Neuron *> all;

    size_t cycleIndex = 0;
    size_t inferenceIndex = 0;

    size_t maxCycle = 200;
    size_t maxInference = 100;

    double totalError;
    double lastError;

    Network() {
        for (size_t k = 0; k < 10; k++) {
            const auto n = new Neuron(1, 1, true);
            inputs.push_back(n);
            n->name = "input";
        }
        for (size_t k = 0; k < 2; k++) {
            const auto n = new Neuron(1, 1, true);
            biases.push_back(n);
            Matrix d(1, 1);
            d(0, 0) = 1.f;
            n->setValues(d);
            n->name = "bias";
        }
        outputs.at(0)->name = "output";

        for (size_t i = 0; i < 10; i++) {
            neurons.push_back(new Neuron(1, 1));
            neurons.at(neurons.size() - 1)->name = "n" + std::to_string(i);
        }
        for (size_t i = 0; i < neurons.size(); i++) {
            for (size_t j = 0; j < neurons.size(); j++) {
                if (i == j) {
                    continue;
                }
                neurons.at(i)->addPossibleConnection(neurons.at(j));
                neurons.at(j)->addPossibleConnection(neurons.at(i));
            }
            for (size_t j = 0; j < inputs.size(); j++) {
                inputs.at(j)->addPossibleConnection(neurons.at(i));
                neurons.at(i)->addPossibleConnection(inputs.at(j));
            }
            for (size_t j = 0; j < outputs.size(); j++) {
                outputs.at(j)->addPossibleConnection(neurons.at(i));
                neurons.at(i)->addPossibleConnection(outputs.at(j));
            }
            for (size_t j = 0; j < biases.size(); j++) {
                biases.at(j)->addPossibleConnection(neurons.at(i));
                neurons.at(i)->addPossibleConnection(biases.at(j));
            }
        }
        all.insert(all.end(), neurons.begin(), neurons.end());
        all.insert(all.end(), inputs.begin(), inputs.end());
        all.insert(all.end(), outputs.begin(), outputs.end());
        all.insert(all.end(), biases.begin(), biases.end());
        // totalError = inputs.size() + outputs.size() + neurons.size();
    }

    bool inference(size_t index) {
        if (inferenceIndex == maxInference) {
            return true;
        }
        std::set<Neuron *> visited = {};
        std::set<Neuron *> children = {};
        double output1[] = {dataOutput.at(index)};
        for (size_t k = 0; k < inputs.size(); k++) {
            Matrix d(1, 1);
            if (k == dataInput.at(index)) {
                d(0, 0) = 1;
            } else {
                d(0, 0) = 0;
            }
            inputs.at(k)->setValues(d);
            children.insert(inputs.at(k));
        }
        outputs.at(0)->setValues(Eigen::Map<Matrix>(output1, 1, 1));
        children.insert(outputs.at(0));
        while (children.size() > 0) {
            auto child = *children.begin();
            visited.insert(child);

            const auto errors = child->update();
            totalError -= errors.oldError.sum();
            totalError += errors.newError.sum();

            for (auto &c : child->outgoing) {
                if (visited.find(c.first) == visited.end()) {
                    children.insert(c.first);
                }
            }
            children.erase(child);
        }

        if ((abs(totalError)) <= 0.0001 || abs(lastError - totalError) <= 0.0) {
            return true;
        }
        lastError = totalError;
        inferenceIndex++;
        return false;
    }

    double update() {
        if (cycleIndex == maxCycle) {
            return totalError;
        }

        if (!inference(indexes[cycleIndex % 10])) {
            return totalError;
        }
        inferenceIndex = 0;

        for (auto &n : all) {
            n->recalcWeights();
        }
        for (auto &n : all) {
            n->reset();
        }
        double currentError = totalError;
        totalError = 0;
        lastError = 1;
        cycleIndex++;
        return currentError;
    }

    void test() {
        for (size_t k = 0; k < 10; k++) {
            for (auto &n : all) {
                n->reset();
            }

            for (size_t i = 0; i < maxInference; i++) {
                std::set<Neuron *> visited = {};
                std::set<Neuron *> children = {};
                for (size_t j = 0; j < inputs.size(); j++) {
                    Matrix d(1, 1);
                    if (j == dataInput.at(k)) {
                        d(0, 0) = 1;
                    } else {
                        d(0, 0) = 0;
                    }
                    inputs.at(j)->setValues(d);
                    children.insert(inputs.at(j));
                }
                for (size_t j = 0; j < outputs.size(); j++) {
                    outputs.at(j)->isInput = false;
                }

                while (children.size() > 0) {
                    auto child = *children.begin();
                    visited.insert(child);
                    child->update();
                    for (auto &c : child->outgoing) {
                        if (visited.find(c.first) == visited.end()) {
                            children.insert(c.first);
                        }
                    }
                    children.erase(child);
                }
            }
            const double res = outputs.at(0)->getActivity().sum() /
                               outputs.at(0)->getActivity().size();
            std::cout << "result=" << res << ", input=" << dataInput.at(k)
                      << ", res=" << (res < 0.5 ? 0 : 1) << ", totalError"
                      << totalError << "\n";
        }
    }
};
void drawLine(sf::RenderWindow &window, sf::Vector2f start, sf::Vector2f end,
              float width, sf::Color color) {
    sf::Vector2f delta = end - start;
    float length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    float angle = std::atan2(delta.y, delta.x) * 180.f / 3.14159f;

    sf::RectangleShape line(sf::Vector2f(length, width));
    line.setPosition(start);
    line.setRotation(angle);
    line.setFillColor(color);
    window.draw(line);
}
enum NeuronType { Output, Input, Normal, Hidden };
struct NeuronObject {
    NeuronType type;
    sf::CircleShape point;
};
int main() {

    Network network = Network();

    for (size_t k = 0; k < network.maxCycle * network.maxInference; k++) {
        network.update();
    }
    network.test();
    return 0;

    std::map<Neuron *, NeuronObject> neuronsElems;

    neuronsElems.insert({network.inputs.at(0),
                         {
                             .type = NeuronType::Input,
                             .point = sf::CircleShape(),
                         }});

    neuronsElems.insert(
        {network.outputs.at(0),
         {.type = NeuronType::Output, .point = sf::CircleShape()}});

    for (auto &n : network.neurons) {
        neuronsElems.insert(
            {n, {.type = NeuronType::Normal, .point = sf::CircleShape()}});
    }

    sf::RenderWindow window(sf::VideoMode(1280, 720), "SFML + ImGui");
    window.setFramerateLimit(60);

    if (!ImGui::SFML::Init(window))
        return -1;
    const float pointSize = 20;
    for (auto &[_, n] : neuronsElems) {
        n.point.setFillColor(n.type == NeuronType::Input    ? sf::Color::Green
                             : n.type == NeuronType::Output ? sf::Color::Red
                                                            : sf::Color::Blue);
        n.point.setPosition(
            sf::Vector2f(window.getSize().x / 2.0f, window.getSize().y / 2.0f));
        n.point.setRadius(pointSize);
    }

    sf::Clock deltaClock;
    sf::Clock updateClock;
    updateClock.restart();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> errorQueue = {};
    std::vector<double> errorX = {};

    while (window.isOpen()) {
        deltaClock.restart();
        float deltaTime = deltaClock.getElapsedTime().asSeconds();
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == sf::Event::Closed)
                window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        ImPlot::CreateContext();

        if (ImPlot::BeginPlot("Total Error")) {
            ImPlot::SetupAxes("X", "Y");
            ImPlot::PlotLine("Error", errorX.data(), errorQueue.data(),
                             errorX.size());
            ImPlot::EndPlot();
        }

        window.clear();
        if (updateClock.getElapsedTime().asSeconds() >= 0.100) {
            network.update();
            std::cout << "update" << "\n";
            updateClock.restart();
        }

        for (auto &[p, n] : neuronsElems) {
            sf::Vector2f pos1 = n.point.getPosition() +
                                n.point.getGlobalBounds().getSize() / 2.f;
            for (auto [n2, c] : p->getPossibleConnections()) {
                auto p = neuronsElems.at(n2).point;
                sf::Vector2f pos2 =
                    p.getPosition() + p.getGlobalBounds().getSize() / 2.f;
                sf::Color m = sf::Color::Magenta;
                m.a = 100;
                drawLine(window, pos1, pos2, c->value * 2, m);
            }
        }
        for (auto &[p, n] : neuronsElems) {
            sf::Vector2f pos1 = n.point.getPosition() +
                                n.point.getGlobalBounds().getSize() / 2.f;
            for (auto [n2, c] : p->outgoing) {
                auto p = neuronsElems.at(n2).point;
                sf::Vector2f pos2 =
                    p.getPosition() + p.getGlobalBounds().getSize() / 2.f;
                sf::Color g = sf::Color::Green;
                g.a = 100;
                drawLine(window, pos1, pos2, c.incoming->value * 3, g);
            }
        }
        for (auto &[p, n] : neuronsElems) {
            window.draw(n.point);

            float dx = n.point.getPosition().x - window.getSize().x / 2.f;
            float dy = n.point.getPosition().y - window.getSize().y / 2.f;

            float angle = std::atan2(-dy, -dx);
            float speed = 0.4f;
            n.point.move(std::cos(angle) * speed, std::sin(angle) * speed);
            for (auto &[_, n2] : neuronsElems) {
                float dx = n.point.getPosition().x - n2.point.getPosition().x;
                float dy = n.point.getPosition().y - n2.point.getPosition().y;

                float dist = sqrt(pow(dx, 2) + pow(dy, 2));
                if (dist == 0.0f) {
                    dx = dis(gen);
                    dy = dis(gen);
                }
                if (dist < pointSize * 4) {
                    float angle = std::atan2(dy, dx);
                    float speed = 1.f;
                    n.point.move(std::cos(angle) * speed,
                                 std::sin(angle) * speed);

                } else {
                    float angle = std::atan2(-dy, -dx);
                    float speed = 0.f;
                    n.point.move(std::cos(angle) * speed,
                                 std::sin(angle) * speed);
                }
            }
        }
        drawLine(window, sf::Vector2f(100, 100), sf::Vector2f(100, 200), 10,
                 sf::Color::Blue);

        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}