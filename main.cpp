#include "inputs/simple_input.hpp"
#include "inputs/xor_input.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <implot.h>
#include <inputs/palindrome_input.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <random>
#include <raylib.h>
#include <rlImGui.h>
#include <set>
#include <string>
#include <vector>

typedef Eigen::MatrixXd Mat;
class Neuron;

bool paused = true;

struct Connection {
    Mat activityWeights;
};

typedef std::shared_ptr<Neuron> neuron_ptr;
typedef std::shared_ptr<Connection> conn_ptr;

struct OutgoingConnection {
    conn_ptr incoming;
};

struct RecalcResponse {
    Mat oldError;
    Mat newError;
};

enum NeuronType { Output, Input, Normal, Bias };

class Neuron : public std::enable_shared_from_this<Neuron> {
  protected:
    Mat activity;
    Mat prediction;
    Mat error;
    Mat sd;

  public:
    NeuronType type;
    bool isInput;
    std::map<neuron_ptr, conn_ptr> incoming = {};
    std::map<neuron_ptr, OutgoingConnection> outgoing = {};
    std::string name;

    Neuron(size_t rows, size_t cols, bool isInput = false,
           NeuronType type = NeuronType::Normal) {
        activity = Mat::Random(rows, cols) * 0.1;
        prediction = Mat::Random(rows, cols) * 0.1;
        error = Mat::Ones(rows, cols);
        this->isInput = isInput;
        this->type = type;
    }

    Mat getActivity() { return this->activity; }

    Mat getPrediction() { return this->prediction; }

    Mat getError() { return this->error; }

    void reset() {
        size_t rows = activity.rows();
        size_t cols = activity.cols();

        activity = Mat::Random(rows, cols) * 0.1;
        prediction = Mat::Random(rows, cols) * 0.1;
    }

    void setActivity(Mat activity) { this->activity = activity; }

    void recalcWeights() {
        for (auto &[neuron, connection] : incoming) {
            connection->activityWeights += this->error.cwiseProduct(
                (this->sd).cwiseProduct(neuron->activity));
        }
    }

    void addConnection(neuron_ptr neuron) {
        const auto &rows = neuron->activity.rows();
        const auto &cols = neuron->activity.cols();

        conn_ptr newConnection = std::make_shared<Connection>(Connection{
            .activityWeights = Mat::Random(rows, cols) * 0.1,
        });
        outgoing.insert({neuron, {.incoming = newConnection}});
        neuron->incoming.insert({shared_from_this(), newConnection});
    }

    Mat sigmoidDerivative(Mat m) {
        auto e = (-m.array()).exp();
        auto res = (1 + e);
        return (e / (res * res)).matrix();
    }

    Mat sigmoid(Mat m) { return (1 / (1 + (-m.array()).exp())).matrix(); }

    void recalcActivity() {
        if (isInput) {
            return;
        }
        Mat delta = -error;

        std::vector<neuron_ptr> toErase = {};

        for (const auto &[neuron, connection] : outgoing) {
            const auto thisWeights = connection.incoming->activityWeights;

            delta += thisWeights.cwiseProduct(
                neuron->error.cwiseProduct(neuron->sd));
        }
        this->activity += delta * 0.1;
    }

    RecalcResponse update() {
        recalcPrediction();
        const Mat oldError = this->error;
        recalcError();
        const Mat newError = this->error;
        recalcActivity();
        return {oldError, newError};
    }

    void recalcPrediction() {
        this->prediction.setZero();
        for (const auto &[neuron, connection] : incoming) {
            this->prediction +=
                neuron->activity.cwiseProduct(connection->activityWeights);
        }
        sd = sigmoidDerivative(this->prediction);
        this->prediction = sigmoid(this->prediction);
    }

    void recalcError() { this->error = (activity - prediction); }
};

struct NetworkResponse {
    double error;
    size_t dataIndex;
};
class Network {
  public:
    std::vector<neuron_ptr> inputs = {};
    std::vector<neuron_ptr> outputs = {};
    std::vector<neuron_ptr> neurons = {};
    std::vector<neuron_ptr> biases = {};
    std::vector<neuron_ptr> all;

    std::shared_ptr<class Input> input;

    size_t cycleIndex = 0;
    size_t inferenceIndex = 0;

    size_t maxCycle = 5000;
    size_t maxInference = 200;

    double totalError;
    double lastError = 0;
    void save(char *filename) {
        std::ofstream file(filename);
        for (auto &n : neurons) {
            for (auto &[n2, c] : n->incoming) {
                file << c->activityWeights(0, 0) << "\n";
            }
        }
        for (auto &n : outputs) {
            for (auto &[n2, c] : n->incoming) {
                file << c->activityWeights(0, 0) << "\n";
            }
        }
    }

    void load(char *filename) {
        std::ifstream file(filename);
        for (auto &n : neurons) {
            for (auto &[n2, c] : n->incoming) {
                file >> c->activityWeights(0, 0);
                std::cout << c->activityWeights(0, 0) << "\n";
            }
        }

        for (auto &n : outputs) {
            for (auto &[n2, c] : n->incoming) {
                file >> c->activityWeights(0, 0);
                std::cout << c->activityWeights(0, 0) << "\n";
            }
        }
    }

    Network(std::shared_ptr<class Input> _input) {
        this->input = _input;
        for (size_t k = 0; k < input->inputSize; k++) {
            const auto n =
                std::make_shared<Neuron>(1, 1, true, NeuronType::Input);
            inputs.push_back(n);
            n->name = "input";
        }

        for (size_t k = 0; k < input->outputSize; k++) {
            const auto n =
                std::make_shared<Neuron>(1, 1, true, NeuronType::Output);
            outputs.push_back(n);
            n->name = "output";
        }

        for (size_t i = 0; i < 2; i++) {
            const auto n =
                std::make_shared<Neuron>(1, 1, false, NeuronType::Normal);
            n->name = "n" + std::to_string(i);
            neurons.push_back(n);
        }
        for (size_t k = 0; k < neurons.size(); k++) {
            const auto n =
                std::make_shared<Neuron>(1, 1, true, NeuronType::Bias);
            biases.push_back(n);
            Mat d(1, 1);
            d(0, 0) = 1.f;
            n->setActivity(d);
            n->name = "bias";
        }
        for (size_t i = 0; i < neurons.size(); i++) {
            for (size_t j = 0; j < inputs.size(); j++) {
                inputs.at(j)->addConnection(neurons.at(i));
            }
            for (size_t j = 0; j < outputs.size(); j++) {
                neurons.at(i)->addConnection(outputs.at(j));
            }
            biases.at(i)->addConnection(neurons.at(i));
        }
        for (auto &n : outputs) {
            neuron_ptr bias =
                std::make_shared<Neuron>(1, 1, true, NeuronType::Bias);
            biases.push_back(bias);
            bias->addConnection(n);
        }
        all.insert(all.end(), neurons.begin(), neurons.end());
        all.insert(all.end(), inputs.begin(), inputs.end());
        all.insert(all.end(), outputs.begin(), outputs.end());
        all.insert(all.end(), biases.begin(), biases.end());
    }

    bool inference(size_t index) {
        if (inferenceIndex == maxInference) {
            return true;
        }
        for (size_t k = 0; k < inputs.size(); k++) {
            inputs.at(k)->setActivity(input->inputSetup(k, index));
        }
        for (size_t k = 0; k < outputs.size(); k++) {
            outputs.at(k)->isInput = true;
            outputs.at(k)->setActivity(input->outputSetup(k, index));
        }

        for (auto &n : all) {
            n->recalcPrediction();
        }

        std::set<neuron_ptr> visited(outputs.begin(), outputs.end());
        std::set<neuron_ptr> children(outputs.begin(), outputs.end());

        while (children.size() > 0) {
            std::set<neuron_ptr> newChildren;
            for (const auto &child : children) {
                totalError -= abs(child->getError()(0, 0));
                const auto errors = child->update();
                totalError += abs(child->getError()(0, 0));
                for (const auto &c : child->incoming) {
                    if (visited.find(c.first) == visited.end()) {
                        visited.insert(c.first);
                        newChildren.insert(c.first);
                    }
                }
            }
            children = newChildren;
        }
        if ((abs(totalError)) <= 0.0001 || abs(lastError - totalError) <= 0.0) {
            return true;
        }
        lastError = totalError;
        inferenceIndex++;
        return false;
    }

    NetworkResponse update() {
        if (cycleIndex >= maxCycle) {
            cycleIndex++;
            return {.error = lastError,
                    .dataIndex = maxCycle % input->dataCount};
        }

        if (!inference(cycleIndex)) {
            return {.error = lastError,
                    .dataIndex = cycleIndex % input->dataCount};
        }
        inferenceIndex = 0;

        double currentError = totalError;
        totalError = 0;
        for (auto &n : all) {
            n->recalcWeights();
        }
        for (auto &n : all) {
            n->reset();
            if (n->type != NeuronType::Bias && n->type != NeuronType::Input) {
                totalError += abs(n->getError().sum());
            }
        }
        for (auto &b : biases) {
            Mat d(1, 1);
            d(0, 0) = 1.f;
            b->setActivity(d);
        }

        lastError = 0;
        cycleIndex++;
        return {.error = currentError,
                .dataIndex = (cycleIndex - 1) % input->dataCount};
    }

    void test() {
        std::cout << "\n\n";
        for (size_t k = 0; k < input->dataCount; k++) {
            totalError = 0;
            for (auto &n : all) {
                n->reset();
            }
            for (size_t j = 0; j < inputs.size(); j++) {
                inputs.at(j)->setActivity(input->inputSetup(j, k));
            }
            for (size_t j = 0; j < outputs.size(); j++) {
                outputs.at(j)->isInput = false;
            }
            for (size_t k = 0; k < biases.size(); k++) {
                Mat d(1, 1);
                d(0, 0) = 1.f;
                biases.at(k)->setActivity(d);
            }
            for (auto &n : all) {
                n->recalcPrediction();
                if (n->type != NeuronType::Bias &&
                    n->type != NeuronType::Input) {
                    totalError += abs(n->getError().sum());
                }
            }

            lastError = 1;
            for (size_t i = 0; i < maxInference; i++) {
                std::set<neuron_ptr> visited(inputs.begin(), inputs.end());
                std::set<neuron_ptr> children(inputs.begin(), inputs.end());
                while (children.size() > 0) {
                    std::set<neuron_ptr> newChildren;
                    for (const auto &child : children) {
                        totalError -= abs(child->getError()(0, 0));
                        child->update();
                        totalError += abs(child->getError()(0, 0));

                        for (const auto &c : child->outgoing) {
                            if (visited.find(c.first) == visited.end()) {
                                visited.insert(c.first);
                                newChildren.insert(c.first);
                            }
                        }
                    }
                    children = newChildren;
                }
            }
            const double res = outputs.at(0)->getPrediction().sum();
            std::cout << std::fixed << std::setprecision(4)
                      << "input=" << input->getInputString(k)
                      << ", result=" << res << ", res=" << (res < 0.5 ? 0 : 1)
                      << ", totalError=" << totalError << "\n";
        }
    }
};
void drawLine(Vector2 start, Vector2 end, float widthStart, float widthEnd,
              Color color) {
    const Vector2 delta = {end.x - start.x, end.y - start.y};
    const float length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (length == 0.0f)
        return;

    const Vector2 perp = {-delta.y / length, delta.x / length};
    const float hwStart = widthStart * -0.5f;
    const float hwEnd = widthEnd * -0.5f;

    const Vector2 vertices[4] = {
        {start.x + perp.x * hwStart, start.y + perp.y * hwStart},
        {start.x - perp.x * hwStart, start.y - perp.y * hwStart},
        {end.x + perp.x * hwEnd, end.y + perp.y * hwEnd},
        {end.x - perp.x * hwEnd, end.y - perp.y * hwEnd}};

    DrawTriangleStrip(vertices, 4, color);
}
struct NeuronObject {
    NeuronType type;
    Vector2 pos;
    float size;
    Color color;
};
int main() {

    Network network = Network(std::make_shared<XorInput>());

    std::map<neuron_ptr, NeuronObject> neuronsElems;

    for (auto &n : network.outputs) {
        neuronsElems.insert({n, {.type = NeuronType::Output}});
    }
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
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    const float pointSize = 20;
    const float markerSize = 3;
    const float outlineSize = pointSize * 1.2;
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

    std::vector<std::vector<double>> errorQueue = {{}};
    std::vector<std::vector<double>> inferenceErrors = {{}};
    std::vector<double> inferenceX = {};
    for (double i = 0; i < network.maxInference; i++) {
        inferenceX.push_back((i + 1) / network.maxInference);
    }
    std::vector<double> inferenceMovingX = {};
    std::vector<std::vector<double>> errorX = {};

    for (size_t i = 0; i < network.input->dataCount; i++) {
        errorQueue.push_back({});
        errorX.push_back({});
    }
    double maxError = 0;

    neuron_ptr selected;
    bool isMouseDown = false;
    bool showOutgoing = true;
    bool showIncoming = false;
    bool showRepercussion = true;
    bool toSave = false;
    char saveFilename[100] = "./weights2";
    char loadFilename[100] = "./weights2";
    while (!WindowShouldClose()) {
        BeginDrawing();
        rlImGuiBegin();
        const bool input = ImGui::GetIO().WantCaptureMouse;

        ImGui::Begin("TotalError");
        if (ImPlot::BeginPlot("##totalerror", {-1, 120},
                              ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxis(ImAxis_X1, "",
                              ImPlotAxisFlags_NoTickLabels |
                                  ImPlotAxisFlags_NoTickMarks);
            ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f");
            ImPlot::SetupAxesLimits(0, 1, 0, 1, ImPlotCond_Always);

            for (size_t i = 0; i < network.input->dataCount; i++) {
                ImPlotSpec spec;
                spec.LineWeight = 2;
                ImVec4 color =
                    ImPlot::GetColormapColor(i % ImPlot::GetColormapSize());
                color.w = 0.6;
                spec.LineColor = color;
                ImPlot::PlotLine(std::to_string(i).c_str(), errorX.at(i).data(),
                                 errorQueue.at(i).data(), errorX.at(i).size(),
                                 spec);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Inference");
        if (ImPlot::BeginPlot("##inference", {-1, 120}, ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxis(ImAxis_X1, "",
                              ImPlotAxisFlags_NoTickLabels |
                                  ImPlotAxisFlags_NoTickMarks);
            ImPlot::SetupAxesLimits(0, 1, 0, 2, ImPlotCond_Always);
            ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f");

            size_t stride = (float)(inferenceErrors.size()) /
                            ((float)inferenceErrors.size() / 50.0);
            for (size_t i = 0; i < inferenceErrors.size() - 1; i += stride) {
                ImPlotSpec spec;
                float v = (float)i / inferenceErrors.size();
                spec.LineColor = ImVec4(0.4 * v, 0.5 * v, 1 * v, 1);
                ImPlot::PlotLine("Inference", inferenceX.data(),
                                 inferenceErrors.at(i).data(),
                                 inferenceErrors.at(i).size(), spec);
            }

            size_t infIndex = inferenceErrors.size() - 1;
            ImPlot::PlotLine("Inference", inferenceMovingX.data(),
                             inferenceErrors.at(infIndex).data(),
                             inferenceErrors.at(infIndex).size());
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Options");
        if (ImGui::Button(paused ? "Go" : "Stop", {50, 30})) {
            paused = !paused;
        }
        ImGui::Checkbox("Show Outgoing", &showOutgoing);
        ImGui::Checkbox("Show Incoming", &showIncoming);

        ImGui::BeginGroup();
        if (ImGui::Button("Save")) {
            toSave = true;
        }
        ImGui::SameLine();

        if (toSave) {
            if (ImGui::Button("Cancel")) {
                toSave = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Confirm")) {
                network.save(saveFilename);
                toSave = false;
            }
        } else {
            ImGui::InputText("File##savefile", saveFilename, 100);
        }
        ImGui::EndGroup();
        ImGui::BeginGroup();
        if (ImGui::Button("Load")) {
            network.load(loadFilename);
        }
        ImGui::SameLine();
        ImGui::InputText("File##loadfile", loadFilename, 100);
        ImGui::EndGroup();
        if (ImGui::Button("Test")) {
            network.test();
        }
        ImGui::End();

        ImGui::Begin("Info");
        if (selected != nullptr) {
            ImGui::Text("Neuron: %s", selected->name.c_str());
            ImGui::Text("Activity: %f", selected->getActivity().sum());
            ImGui::Text("Prediction: %f", selected->getPrediction().sum());
            ImGui::Text("Error: %f", (selected->getError().cwiseAbs()).sum());
            ImGui::Checkbox("Show Repercussion", &showRepercussion);
        } else {
            ImGui::Text("Click on a neuron to see its info");
        }
        ImGui::End();

        if (!input && ImGui::IsMouseDown(MOUSE_LEFT_BUTTON)) {
            if (isMouseDown && selected != nullptr) {
                auto &n = neuronsElems.at(selected);
                n.pos = GetMousePosition();
            }
        }
        if (!input &&
            ImGui::IsMouseClicked(MOUSE_LEFT_BUTTON, ImGuiInputFlags_None)) {
            const auto pos = ImGui::GetMousePos();
            neuron_ptr newSelected = nullptr;
            for (const auto &[p, n] : neuronsElems) {
                auto d = Vector2{n.pos.x - pos.x, n.pos.y - pos.y};
                auto dist = sqrt(pow(d.x, 2) + pow(d.y, 2));
                if (dist <= n.size) {
                    newSelected = p;
                }
            }
            selected = newSelected;
            isMouseDown = true;
        }
        if (!input && ImGui::IsMouseReleased(MOUSE_BUTTON_LEFT)) {
            isMouseDown = false;
        }
        if (!paused) {
            for (size_t k = 0; k < 10; k++) {
                if (paused)
                    break;
                const NetworkResponse res = network.update();
                if (network.inferenceIndex == network.maxInference) {
                    errorQueue.at(res.dataIndex).push_back(res.error);
                    errorX.at(res.dataIndex)
                        .push_back(
                            ((double)(errorQueue.at(res.dataIndex).size())) /
                            ((double)network.maxCycle /
                             network.input->dataCount));
                    inferenceErrors.push_back({});
                    inferenceMovingX = {};
                    continue;
                }
                inferenceErrors.at(inferenceErrors.size() - 1)
                    .push_back(res.error);
                inferenceMovingX.push_back(
                    (double)(inferenceErrors.at(inferenceErrors.size() - 1)
                                 .size()) /
                    network.maxInference);
            }
        }
        ClearBackground(BLACK);

        std::vector<Vector2> outgoingMarkers;
        for (auto &[p, n] : neuronsElems) {
            if (selected != nullptr && p != selected) {
                continue;
            }

            const Vector2 pos1 = {n.pos.x, n.pos.y};

            if (selected != nullptr && showRepercussion) {
                std::set<neuron_ptr> visited = {};
                std::set<neuron_ptr> children = {p};
                while (!children.empty()) {
                    std::set<neuron_ptr> newChildren;
                    for (const auto &child : children) {
                        const auto &childObj = neuronsElems.at(child);
                        Vector2 pos1 = {childObj.pos.x, childObj.pos.y};
                        for (const auto &[n2, c] : child->outgoing) {
                            if (visited.find(n2) != visited.end()) {
                                continue;
                            }
                            newChildren.insert(n2);
                            auto obj = neuronsElems.at(n2);
                            const auto dist =
                                Vector2{obj.pos.x - childObj.pos.x,
                                        obj.pos.y - childObj.pos.y};
                            const double angle = std::atan2(dist.y, dist.x);
                            Vector2 pointPos = Vector2{
                                (float)(cos(angle) * (pointSize + markerSize)),
                                (float)(sin(angle) * (pointSize + markerSize))};
                            outgoingMarkers.push_back(
                                Vector2{childObj.pos.x + pointPos.x,
                                        childObj.pos.y + pointPos.y});
                            Vector2 pos2 = {obj.pos.x, obj.pos.y};
                            Color m = c.incoming->activityWeights.sum() < 0
                                          ? BLUE
                                          : GREEN;
                            double width =
                                abs(c.incoming->activityWeights.sum()) * 4;
                            m.a = 100;
                            drawLine(pos1, pos2, width, width, m);
                        }
                    }
                    visited.insert(children.begin(), children.end());
                    children = newChildren;
                }
            }
            if (showOutgoing)
                for (const auto &[n2, c] : p->outgoing) {
                    auto p = neuronsElems.at(n2);
                    Vector2 pos2 = {p.pos.x, p.pos.y};
                    Color m =
                        c.incoming->activityWeights.sum() < 0 ? BLUE : GREEN;
                    double width = abs(c.incoming->activityWeights.sum()) * 4;
                    m.a = 100;
                    drawLine(pos1, pos2, width, width, m);
                }
            if (showIncoming)
                for (const auto &[n2, c] : p->incoming) {
                    auto p = neuronsElems.at(n2);
                    Vector2 pos2 = {p.pos.x, p.pos.y};
                    Color m = c->activityWeights.sum() < 0 ? BLUE : GREEN;
                    double width = abs(c->activityWeights.sum()) * 4;
                    m.a = 100;
                    drawLine(pos1, pos2, width, width, m);
                }
        }

        for (const auto &marker : outgoingMarkers) {
            DrawCircle(marker.x, marker.y, markerSize, WHITE);
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
                if (dist < pointSize * 4) {
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

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
