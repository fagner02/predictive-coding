#include "inputs/simple_input.hpp"
#include "network_c.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <implot.h>
#include <inputs/palindrome_input.hpp>
#include <random>
#include <raylib.h>
#include <rlImGui.h>
#include <vector>

bool paused = true;

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

    NetworkC network = NetworkC();

    std::map<capped_neuron_ptr, NeuronObject> neuronsElems;

    for (auto &n : network.output) {
        neuronsElems.insert({n, {.type = NeuronType::Output}});
    }
    for (auto &n : network.input) {
        neuronsElems.insert({n, {.type = NeuronType::Input}});
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

    std::vector<double> inferenceMovingX = {};
    std::vector<std::vector<double>> errorX = {};

    double maxError = 0;

    capped_neuron_ptr selected;
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
        // if (ImPlot::BeginPlot("##totalerror", {-1, 120},
        // ImPlotFlags_NoLegend)) {
        //     ImPlot::SetupAxis(ImAxis_X1, "",
        //                       ImPlotAxisFlags_NoTickLabels
        //                       |
        //                           ImPlotAxisFlags_NoTickMarks);
        //     ImPlot::SetupAxisFormat(ImAxis_Y1,
        //     "%.1f"); ImPlot::SetupAxesLimits(0, 1,
        //     0, 1, ImPlotCond_Always);

        //     for (size_t i = 0; i <
        //     network.input->dataCount; i++) {
        //         ImPlotSpec spec;
        //         spec.LineWeight = 2;
        //         ImVec4 color =
        //             ImPlot::GetColormapColor(i %
        //             ImPlot::GetColormapSize());
        //         color.w = 0.6;
        //         spec.LineColor = color;
        //         ImPlot::PlotLine(std::to_string(i).c_str(),
        //         errorX.at(i).data(),
        //                          errorQueue.at(i).data(),
        //                          errorX.at(i).size(),
        //                          spec);
        //     }
        //     ImPlot::EndPlot();
        // }
        ImGui::End();

        ImGui::Begin("Inference");
        // if (ImPlot::BeginPlot("##inference", {-1, 120},
        // ImPlotFlags_NoLegend)) {
        //     ImPlot::SetupAxis(ImAxis_X1, "",
        //                       ImPlotAxisFlags_NoTickLabels |
        //                           ImPlotAxisFlags_NoTickMarks);
        //     ImPlot::SetupAxesLimits(0, 1, 0, 2, ImPlotCond_Always);
        //     ImPlot::SetupAxisFormat(ImAxis_Y1, "%.1f");

        //     size_t stride = (float)(inferenceErrors.size()) /
        //                     ((float)inferenceErrors.size() / 50.0);
        //     for (size_t i = 0; i < inferenceErrors.size() - 1; i += stride) {
        //         ImPlotSpec spec;
        //         float v = (float)i / inferenceErrors.size();
        //         spec.LineColor = ImVec4(0.4 * v, 0.5 * v, 1 * v, 1);
        //         ImPlot::PlotLine("Inference", inferenceX.data(),
        //                          inferenceErrors.at(i).data(),
        //                          inferenceErrors.at(i).size(), spec);
        //     }

        //     size_t infIndex = inferenceErrors.size() - 1;
        //     ImPlot::PlotLine("Inference", inferenceMovingX.data(),
        //                      inferenceErrors.at(infIndex).data(),
        //                      inferenceErrors.at(infIndex).size());
        //     ImPlot::EndPlot();
        // }
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
                // network.save(saveFilename);
                toSave = false;
            }
        } else {
            ImGui::InputText("File##savefile", saveFilename, 100);
        }
        ImGui::EndGroup();
        ImGui::BeginGroup();
        if (ImGui::Button("Load")) {
            // network.load(loadFilename);
        }
        ImGui::SameLine();
        ImGui::InputText("File##loadfile", loadFilename, 100);
        ImGui::EndGroup();
        if (ImGui::Button("Test")) {
            // network.test();
        }
        ImGui::End();

        ImGui::Begin("Info");
        if (selected != nullptr) {
            // ImGui::Text("Neuron: %s", selected->name.c_str());
            ImGui::Text("Activity: %f", selected->activity);
            ImGui::Text("Lower Bound: %f", selected->lowerBound);
            ImGui::Text("Is Active: %b", selected->isActive);
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
            capped_neuron_ptr newSelected = nullptr;
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
            network.update();
        }
        ClearBackground(BLACK);

        std::vector<Vector2> outgoingMarkers;
        for (auto &[p, n] : neuronsElems) {
            if (selected != nullptr && p != selected) {
                continue;
            }

            const Vector2 pos1 = {n.pos.x, n.pos.y};

            if (selected != nullptr && showRepercussion) {
                std::set<capped_neuron_ptr> visited = {};
                std::set<capped_neuron_ptr> children = {p};
                while (!children.empty()) {
                    std::set<capped_neuron_ptr> newChildren;
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
                            Color m = c.connection->weight < 0 ? BLUE : GREEN;
                            double width = abs(c.connection->weight) * 4;
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
                    Color m = c.connection->weight < 0 ? BLUE : GREEN;
                    double width = abs(c.connection->weight) * 4;
                    m.a = 100;
                    drawLine(pos1, pos2, width, width, m);
                }
            if (showIncoming)
                for (const auto &[n2, c] : p->incoming) {
                    auto p = neuronsElems.at(n2);
                    Vector2 pos2 = {p.pos.x, p.pos.y};
                    Color m = c->weight < 0 ? BLUE : GREEN;
                    double width = abs(c->weight) * 4;
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
