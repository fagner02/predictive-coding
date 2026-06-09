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

double totalError = 0;
const double connectionValueLimit = 2;
class Neuron {
  protected:
    std::map<Neuron *, Connection *> possibleConnections;
    Matrix activity;
    Matrix prediction;
    Matrix error;
    Matrix z;
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
            const Matrix sd = sigmoidDerivative(z);
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

    void recalcActivity() {
        recalcPrediction();
        recalcError();
        if (!isInput) {

            Matrix delta = -error;

            for (const auto &[neuron, connection] : outgoing) {
                const auto thisWeights =
                    connection.incoming->actual.activityWeights;
                delta += thisWeights.cwiseProduct(
                    neuron->error.cwiseProduct(neuron->sd));

                Matrix ta = sigmoid(this->activity.cwiseProduct(
                    connection.incoming->actual.activityWeights));

                const auto res =
                    ta.cwiseProduct(connection.incoming->actual.activityWeights)
                        .cwiseProduct(neuron->error);
                const double avg = res.sum();
                connection.incoming->value += avg;
                connection.incoming->value =
                    connection.incoming->value > connectionValueLimit
                        ? connectionValueLimit
                    : connection.incoming->value < -connectionValueLimit
                        ? -connectionValueLimit
                        : connection.incoming->value;
            }
            this->activity += delta;
        }

        std::vector<Neuron *> toErase = {};
        for (auto &[neuron, posCon] : possibleConnections) {
            Matrix ta = sigmoid(
                this->activity.cwiseProduct(posCon->actual.activityWeights));

            const auto res = ta.cwiseProduct(posCon->actual.activityWeights)
                                 .cwiseProduct(neuron->error);
            const double avg = res.sum();
            posCon->value += avg;
            posCon->value =
                posCon->value > connectionValueLimit    ? connectionValueLimit
                : posCon->value < -connectionValueLimit ? -connectionValueLimit
                                                        : posCon->value;

            if (posCon->value > 0.5) {
                toErase.push_back(neuron);
                neuron->incoming.insert({this, new Connection(*posCon)});
                outgoing.insert(
                    {neuron, OutgoingConnection{
                                 .incoming = neuron->incoming.at(this)}});

                // incoming.insert({neuron, new Connection(*posCon)});
                // neuron->outgoing.insert(
                //     {this,
                //      OutgoingConnection{.incoming = incoming.at(neuron)}});

                std::cout << "connection executed" << this->name << "->"
                          << neuron->name << "\n";
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
            this->prediction += neuron->activity.cwiseProduct(
                connection->actual.activityWeights);
        }
        sd = sigmoidDerivative(this->prediction);
        this->prediction = sigmoid(this->prediction);
    }

    void recalcError() {
        totalError -= this->error.cwiseAbs().sum() / this->error.size();
        this->error = (activity - prediction);
        totalError += this->error.cwiseAbs().sum() / this->error.size();
    }
};

std::vector<double> dataInput = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<double> dataOutput = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
// std::vector<double> dataOutput = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
const size_t indexes[] = {7, 0, 5, 9, 3, 2, 4, 1, 6, 8};

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

class Network {
  public:
    std::vector<Neuron *> neurons = {};
    std::vector<Neuron *> inputs = {new Neuron(1, 1, true)};
    std::vector<Neuron *> outputs = {new Neuron(1, 1, true)};
    size_t maxPass = 500;
    size_t maxCycle = 100;
    size_t pass = 0;
    size_t cycle = 0;

    double lastError = 1;
    double totalError = 0;
    Network() {
        inputs.at(0)->name = "input";
        outputs.at(0)->name = "output";

        for (size_t i = 0; i < 10; i++) {
            neurons.push_back(new Neuron(1, 1));
            neurons.at(neurons.size() - 1)->name = "n" + std::to_string(i);
        }
        for (size_t i = 0; i < 10; i++) {
            for (size_t j = 0; j < 10; j++) {
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
        }
    }

    void update() {
        if (pass == maxPass)
            return;
        if (cycle == maxCycle) {
            cycle = 0;
            return;
        }
        const size_t index = indexes[pass % 10];
        for (size_t j = 0; j < inputs.size(); j++) {
            std::set<Neuron *> visited = {};
            std::set<Neuron *> children = {};
            double input[] = {dataInput.at(index) / 10.0};
            double output1[] = {dataOutput.at(index)};
            inputs.at(j)->setValues(Eigen::Map<Matrix>(input, 1, 1));
            outputs.at(0)->setValues(Eigen::Map<Matrix>(output1, 1, 1));
            children.insert(inputs.at(j));
            children.insert(outputs.at(0));

            while (children.size() > 0.2) {
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

        if ((abs(totalError)) <= 0.001 || abs(lastError - totalError) <= 0.0) {
            pass++;
            cycle++;
            lastError = totalError;
            return;
        }
        for (size_t j = 0; j < neurons.size(); j++) {
            neurons.at(j)->recalcWeights();
        }
        for (size_t j = 0; j < inputs.size(); j++) {
            inputs.at(j)->recalcWeights();
        }
        for (size_t j = 0; j < outputs.size(); j++) {
            outputs.at(j)->recalcWeights();
        }
        for (size_t j = 0; j < neurons.size(); j++) {
            neurons.at(j)->reset();
        }
        for (size_t j = 0; j < inputs.size(); j++) {
            inputs.at(j)->reset();
        }
        for (size_t j = 0; j < outputs.size(); j++) {
            outputs.at(j)->reset();
        }
        totalError = 0;
        lastError = 1;
        pass++;
        cycle++;
    }
};

enum NeuronType { Output, Input, Normal, Hidden };
struct NeuronObject {
    NeuronType type;
    sf::CircleShape point;
};
int main() {

    Network network = Network();
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

    // {
    //     for (size_t k = 0; k < 10; k++) {
    //         for (size_t j = 0; j < neurons.size(); j++) {
    //             neurons.at(j)->reset();
    //         }
    //         for (size_t j = 0; j < inputs.size(); j++) {
    //             inputs.at(j)->reset();
    //         }
    //         for (size_t j = 0; j < outputs.size(); j++) {
    //             outputs.at(j)->reset();
    //         }

    //         for (size_t i = 0; i < 100; i++) {
    //             std::set<Neuron *> visited = {};
    //             std::set<Neuron *> children = {};
    //             for (size_t j = 0; j < inputs.size(); j++) {
    //                 double data[] = {dataInput.at(k) / 10.0};
    //                 inputs.at(j)->setValues(Eigen::Map<Matrix>(data, 1,
    //                 1)); children.insert(inputs.at(j));
    //             }
    //             for (size_t j = 0; j < outputs.size(); j++) {
    //                 outputs.at(j)->isInput = false;
    //             }

    //             while (children.size() > 0) {
    //                 auto child = *children.begin();
    //                 visited.insert(child);
    //                 child->recalcActivity();
    //                 for (auto &c : child->outgoing) {
    //                     if (visited.find(c.first) == visited.end()) {
    //                         children.insert(c.first);
    //                     }
    //                 }
    //                 children.erase(child);
    //             }
    //         }
    //         const double res = outputs.at(0)->getActivity().sum() /
    //                            outputs.at(0)->getActivity().size();
    //         std::cout << "result: " << res << ",  " << dataInput.at(k) <<
    //         ",
    //         "
    //                   << (res < 0.5 ? 0 : 1) << ", " << totalError <<
    //                   "\n";
    //     }
    // }
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

        // Your ImGui UI
        ImGui::Begin("Hello");
        // for (size_t i = 0; i < inputs.size(); i++) {

        // ImGui::Text(inputs.at(i)->name+(inputs.at()));
        // }
        ImGui::Text("SFML");
        ImGui::End();

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

            float angle = std::atan2(-dy, -dx); // radians
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

                    // Normalize (unit vector in the direction from n2 → n)
                    float angle = std::atan2(dy, dx); // radians
                    float speed = 1.f;
                    n.point.move(std::cos(angle) * speed,
                                 std::sin(angle) * speed);

                } else {
                    float angle = std::atan2(-dy, -dx); // radians
                    float speed = 0.f;
                    n.point.move(std::cos(angle) * speed,
                                 std::sin(angle) * speed);
                }
            }
        }
        drawLine(window, sf::Vector2f(100, 100), sf::Vector2f(100, 200), 10,
                 sf::Color::Blue);
        // Draw your SFML stuff here
        ImGui::SFML::Render(window); // Render ImGui on top
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}