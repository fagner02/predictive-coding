#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <implot.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <raylib.h>
#include <rlImGui.h>
#include <set>
#include <string>
#include <vector>

typedef Eigen::MatrixXd Mat;

class Neuron;

struct Convert {
    Mat outer;
    Mat inner;
};
struct ProtoConnection {
    Mat activityWeights;
};

struct Connection {
    ProtoConnection actual;
    double value;
};

struct OutgoingConnection {
    Connection *incoming;
};

const double connectionValueLimit = 5;

struct RecalcResponse {
    Mat oldError;
    Mat newError;
};

class Neuron {
  protected:
    std::map<Neuron *, Connection *> possibleConnections;
    Mat activity;
    Mat prediction;
    Mat error;
    Mat sd;

  public:
    bool isInput;
    std::map<Neuron *, Connection *> incoming = {};
    std::map<Neuron *, OutgoingConnection> outgoing = {};
    std::string name;

    Neuron(size_t rows, size_t cols, bool isInput = false) {
        activity = Mat::Zero(rows, cols);
        prediction = Mat::Ones(rows, cols);
        error = Mat::Ones(rows, cols);
        this->isInput = isInput;
    }

    Mat getActivity() { return this->activity; }

    Mat getPrediction() { return this->prediction; }

    void reset() {
        size_t rows = activity.rows();
        size_t cols = activity.cols();

        activity = Mat::Zero(rows, cols);
        prediction = Mat::Ones(rows, cols);
        error = Mat::Ones(rows, cols);
    }

    void setValues(Mat activity) {
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
            .activityWeights = Mat::Random(rows, cols) * 0.1,
        };
        Connection *newPossibleConnection = new Connection{
            .actual = newConnection,
            .value = 0,
        };
        // this -> neuron
        possibleConnections.insert({neuron, newPossibleConnection});
    }

    Mat sigmoidDerivative(Mat m) {
        auto e = (-m.array()).exp();
        auto res = (1 + e);
        return (e / (res * res)).matrix();
    }

    Mat sigmoid(Mat m) { return (1 / (1 + (-m.array()).exp())).matrix(); }

    void recalcConnectionValue(Neuron *neuron, Connection *connection) {
        Mat ta = sigmoid(
            this->activity.cwiseProduct(connection->actual.activityWeights));

        const auto res = ta.cwiseProduct(neuron->error);

        connection->value += res.sum() * 0.1;
        connection->value =
            connection->value > connectionValueLimit    ? connectionValueLimit
            : connection->value < -connectionValueLimit ? -connectionValueLimit
                                                        : connection->value;
    }

    void recalcActivity() {
        if (!isInput) {

            Mat delta = -error;

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

    void recalcPossibleConnections() {
        std::vector<Neuron *> toErase = {};
        for (auto &[neuron, posCon] : possibleConnections) {

            recalcConnectionValue(neuron, posCon);

            // std::cout
            if ((posCon->value) > 0) {
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
    }

    RecalcResponse update() {
        recalcPrediction();
        const Mat oldError = this->error;
        recalcError();
        const Mat newError = this->error;
        recalcActivity();
        recalcPossibleConnections();
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
// std::vector<double> dataOutput = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// std::vector<double> dataOutput = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
std::vector<double> dataOutput = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
// std::vector<double> dataOutput = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1};

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
    double lastError = 0;

    Network() {
        for (size_t k = 0; k < 10; k++) {
            const auto n = new Neuron(1, 1, true);
            inputs.push_back(n);
            n->name = "input";
        }
        for (size_t k = 0; k < 2; k++) {
            const auto n = new Neuron(1, 1, true);
            biases.push_back(n);
            Mat d(1, 1);
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
        totalError =
            inputs.size() + outputs.size() + neurons.size() + biases.size();
    }

    bool inference(size_t index) {
        if (inferenceIndex == maxInference) {
            return true;
        }
        std::set<Neuron *> notVisited(all.begin(), all.end());
        std::set<Neuron *> children = {};
        double output1[] = {dataOutput.at(index)};
        for (size_t k = 0; k < inputs.size(); k++) {
            Mat d(1, 1);
            if (k == dataInput.at(index)) {
                d(0, 0) = 1;
            } else {
                d(0, 0) = 0;
            }
            inputs.at(k)->setValues(d);
            children.insert(inputs.at(k));
        }
        outputs.at(0)->setValues(Eigen::Map<Mat>(output1, 1, 1));
        children.insert(outputs.at(0));
        while (children.size() > 0) {
            auto child = *children.begin();
            notVisited.erase(child);

            const auto errors = child->update();
            totalError -= errors.oldError.sum();
            totalError += errors.newError.sum();

            for (auto &c : child->outgoing) {
                if (notVisited.find(c.first) != notVisited.end()) {
                    children.insert(c.first);
                }
            }
            children.erase(child);
        }
        for (auto child = notVisited.begin(); child != notVisited.end();
             child++) {
            (*child)->recalcPossibleConnections();
        }

        if ((abs(totalError)) <= 0.0001 || abs(lastError - totalError) <= 0.0) {
            return true;
        }
        lastError = totalError;
        inferenceIndex++;
        return false;
    }

    double update() {
        if (cycleIndex >= maxCycle) {
            if (cycleIndex == maxCycle) {
                test();
            }
            cycleIndex++;
            return lastError;
        }

        if (!inference(indexes[cycleIndex % 10])) {
            return lastError;
        }
        inferenceIndex = 0;

        for (auto &n : all) {
            n->recalcWeights();
        }
        for (auto &n : all) {
            n->reset();
        }
        double currentError = totalError;

        totalError = inputs.size() + outputs.size() + neurons.size();
        lastError = 0;
        cycleIndex++;
        return currentError;
    }

    void test() {
        for (size_t k = 0; k < 10; k++) {
            for (auto &n : all) {
                n->reset();
            }

            for (size_t i = 0; i < maxInference; i++) {
                std::set<Neuron *> notVisited =
                    std::set<Neuron *>(all.begin(), all.end());
                std::set<Neuron *> children = {};
                for (size_t j = 0; j < inputs.size(); j++) {
                    Mat d(1, 1);
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
                    notVisited.erase(child);
                    child->update();
                    for (auto &c : child->outgoing) {
                        if (notVisited.find(c.first) != notVisited.end()) {
                            children.insert(c.first);
                        }
                    }
                    children.erase(child);
                }
                for (auto child = notVisited.begin(); child != notVisited.end();
                     child++) {
                    (*child)->recalcPossibleConnections();
                }
            }
            const double res = outputs.at(0)->getActivity().sum();
            std::cout << "result=" << res << ", input=" << dataInput.at(k)
                      << ", res=" << (res < 0.5 ? 0 : 1) << ", totalError"
                      << totalError << "\n";
        }
    }
};
void drawLine(Vector2 start, Vector2 end, float widthStart, float widthEnd,
              Color color) {
    Vector2 delta = {end.x - start.x, end.y - start.y};
    float length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (length == 0.0f)
        return;

    Vector2 perp = {-delta.y / length, delta.x / length};
    float hwStart = widthStart * -0.5f;
    float hwEnd = widthEnd * -0.5f;

    Vector2 vertices[4] = {
        {start.x + perp.x * hwStart, start.y + perp.y * hwStart},
        {start.x - perp.x * hwStart, start.y - perp.y * hwStart},
        {end.x + perp.x * hwEnd, end.y + perp.y * hwEnd},
        {end.x - perp.x * hwEnd, end.y - perp.y * hwEnd}};

    DrawTriangleStrip(vertices, 4, color);
}
enum NeuronType { Output, Input, Normal, Hidden, Bias };

struct NeuronObject {
    NeuronType type;
    Vector2 pos;
    float size;
    Color color;
};
int main() {

    Network network = Network();

    std::map<Neuron *, NeuronObject> neuronsElems;

    neuronsElems.insert({network.outputs.at(0), {.type = NeuronType::Output}});

    for (auto &n : network.inputs) {
        neuronsElems.insert({n, {.type = NeuronType::Input}});
    }
    for (auto &n : network.biases) {
        neuronsElems.insert({n, {.type = NeuronType::Bias}});
    }
    for (auto &n : network.neurons) {
        neuronsElems.insert({n, {.type = NeuronType::Normal}});
    }

    InitWindow(1280, 720, "Title");
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImPlot::CreateContext();

    const float pointSize = 20;
    std::map<NeuronType, Color> colors = {{NeuronType::Normal, BLUE},
                                          {NeuronType::Input, GREEN},
                                          {NeuronType::Output, RED},
                                          {NeuronType::Bias, YELLOW}};
    for (auto &[_, n] : neuronsElems) {
        n.pos = {(float)GetScreenWidth() / 2, (float)GetScreenHeight() / 2};
        n.color = colors.at(n.type);
        n.size = pointSize;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<double> errorQueue = {};
    std::vector<double> errorX = {};
    double maxError = 0;

    Neuron *selected;
    bool isMouseDown = false;

    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();
        if (ImGui::IsMouseDown(MOUSE_LEFT_BUTTON)) {
            if (isMouseDown && selected != nullptr) {
                auto &n = neuronsElems.at(selected);
                n.pos = GetMousePosition();
            }
        }
        if (ImGui::IsMouseClicked(MOUSE_LEFT_BUTTON, ImGuiInputFlags_None)) {
            const auto pos = ImGui::GetMousePos();
            Neuron *newSelected = nullptr;
            for (auto &[p, n] : neuronsElems) {
                auto d = Vector2{n.pos.x - pos.x, n.pos.y - pos.y};
                auto dist = sqrt(pow(d.x, 2) + pow(d.y, 2));
                if (dist <= n.size) {
                    newSelected = p;
                }
            }
            selected = newSelected;
            isMouseDown = true;
        }
        if (ImGui::IsMouseReleased(MOUSE_BUTTON_LEFT)) {
            isMouseDown = false;
        }

        if (ImPlot::BeginPlot("Total Error", {-1, 200})) {
            ImPlot::SetupAxes("X", "Y");
            ImPlot::SetupAxesLimits(0, 20, -maxError * 1.2, maxError * 1.2,
                                    ImPlotCond_Always);
            ImPlot::PlotLine("Error", errorX.data(), errorQueue.data(),
                             errorX.size());
            ImPlot::EndPlot();
        }

        ClearBackground(BLACK);
        for (size_t k = 0; k < 10; k++) {
            const double error = network.update();
            if (errorQueue.size() == 20) {
                errorQueue.erase(errorQueue.begin());
            }
            errorQueue.push_back(error);
            if (maxError < abs(error)) {
                maxError = abs(error);
            }
            if (errorX.size() < 20) {
                errorX = std::vector<double>(errorQueue.size());
                std::iota(errorX.begin(), errorX.end(), 0);
            }
        }

        for (auto &[p, n] : neuronsElems) {
            if (selected != nullptr && p != selected) {
                continue;
            }
            Vector2 pos1 = {n.pos.x, n.pos.y};
            for (auto [n2, c] : p->getPossibleConnections()) {
                auto p = neuronsElems.at(n2);
                Vector2 pos2 = {p.pos.x, p.pos.y};
                Color m = c->value < 0 ? RED : MAGENTA;
                double width = abs(c->value);
                m.a = 100;
                drawLine(pos1, pos2, width, width, m);
            }
        }
        for (auto &[p, n] : neuronsElems) {
            if (selected != nullptr && p != selected) {
                continue;
            }
            Vector2 pos1 = {n.pos.x, n.pos.y};
            for (auto [n2, c] : p->outgoing) {
                auto p = neuronsElems.at(n2);
                Vector2 pos2 = {p.pos.x, p.pos.y};
                Color m = c.incoming->value < 0 ? BLUE : GREEN;
                double width = abs(c.incoming->value);
                m.a = 100;
                drawLine(pos1, pos2, width, width, m);
            }
        }
        for (auto &[p, n] : neuronsElems) {
            if (p == selected) {
                DrawCircle(n.pos.x, n.pos.y, n.size * 1.2, WHITE);
            }
            DrawCircle(n.pos.x, n.pos.y, n.size, n.color);

            const auto windowSize =
                Vector2{(float)GetScreenWidth(), (float)GetScreenHeight()};
            float dx = n.pos.x - windowSize.x / 2.f;
            float dy = n.pos.y - windowSize.y / 2.f;

            float angle = std::atan2(-dy, -dx);
            float speed = 0.4f;
            n.pos = Vector2{n.pos.x + (std::cos(angle) * speed),
                            n.pos.y + std::sin(angle) * speed};
            for (auto &[_, n2] : neuronsElems) {
                float dx = n.pos.x - n2.pos.x;
                float dy = n.pos.y - n2.pos.y;

                float dist = sqrt(pow(dx, 2) + pow(dy, 2));
                if (dist == 0.0f) {
                    dx = dis(gen);
                    dy = dis(gen);
                }
                if (dist < pointSize * 6) {
                    float angle = std::atan2(dy, dx);
                    float speed = 1.f;
                    n.pos = {n.pos.x + std::cos(angle) * speed,
                             n.pos.y + std::sin(angle) * speed};
                }
            }
        }

        rlImGuiEnd();
        EndDrawing();
    }

    ImGui::DestroyContext();
    rlImGuiShutdown();
    CloseWindow();
    return 0;
}