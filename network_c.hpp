#include "capped_neuron.hpp"
#include "types.hpp"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

class NetworkC {
  public:
    std::vector<capped_neuron_ptr> neurons;
    std::vector<capped_neuron_ptr> input;
    std::vector<capped_neuron_ptr> output;

    std::set<capped_neuron_ptr> nextParents = {};

    size_t cycleIndex = 0;
    size_t maxCycle = 100;

    NetworkC() {
        capped_neuron_ptr n =
            std::make_shared<CappedNeuron>(true, NeuronType::Input);
        input.push_back(n);

        for (size_t i = 0; i < 7; i++) {
            n = std::make_shared<CappedNeuron>(false, NeuronType::Normal);
            neurons.push_back(n);
        }

        n = std::make_shared<CappedNeuron>(true, NeuronType::Output);
        output.push_back(n);

        input.at(0)->addConnection(neurons.at(0), ConnectionType::Excitatory);
        neurons.at(0)->addConnection(neurons.at(1), ConnectionType::Excitatory);
        neurons.at(1)->addConnection(neurons.at(2), ConnectionType::Excitatory);
        neurons.at(2)->addConnection(neurons.at(1), ConnectionType::Inhibitory);
        neurons.at(1)->addConnection(neurons.at(3), ConnectionType::Excitatory);
        neurons.at(1)->addConnection(neurons.at(4), ConnectionType::Excitatory);
        neurons.at(3)->addConnection(neurons.at(5), ConnectionType::Excitatory);
        neurons.at(4)->addConnection(neurons.at(6), ConnectionType::Excitatory);
        neurons.at(6)->addConnection(output.at(0), ConnectionType::Excitatory);
        neurons.at(5)->addConnection(output.at(0), ConnectionType::Excitatory);
        neurons.at(3)->addConnection(neurons.at(4), ConnectionType::Inhibitory);
    }

    bool propagate(double data) {
        std::set<capped_neuron_ptr> parents = {nextParents.begin(),
                                               nextParents.end()};
        std::set<capped_neuron_ptr> visited = {};
        input.at(0)->activity = data;
        nextParents = {};

        while (!parents.empty()) {
            std::set<capped_neuron_ptr> newParents = {};
            for (auto &parent : parents) {
                visited.insert(parent);
                if (parent->activity < parent->lowerBound ||
                    parent->activity > parent->upperBound) {
                    continue;
                }
                for (auto &[child, outCon] : parent->outgoing) {
                    child->activity +=
                        parent->activity * outCon.connection->weight;
                    child->isActive = true;
                    if (visited.find(child) == visited.end()) {
                        newParents.insert(child);
                    } else {
                        nextParents.insert(parent);
                    }
                }
            }
            parents = newParents;
        }

        for (auto &n : neurons) {
            if (!n->isActive) {
                continue;
            }
            n->wasActive = true;
            n->isActive = false;

            for (auto &[child, outCon] : n->outgoing) {
                double w = outCon.connection->weight;
                double a = child->activity - n->activity * w;

                double center =
                    child->type == NeuronType::Output
                        ? 1 - (int)data % 2
                        : child->lowerBound +
                              (child->upperBound - child->lowerBound) / 2;

                double increase = a + n->activity * (w + 0.1);
                double decrease = a + n->activity * (w - 0.1);

                bool wasOver = child->activity > child->upperBound;
                bool wasUnder = child->activity < child->lowerBound;

                if (abs(increase - center) < abs(decrease - center)) {
                    if (increase <= child->upperBound || wasOver) {
                        // Should increase
                        outCon.connection->weight = w + 0.1;
                    }
                } else {
                    if (decrease >= child->lowerBound || wasUnder) {
                        // Should decrease
                        outCon.connection->weight = w - 0.1;
                    }
                }
            }
        }
        if (nextParents.empty()) {
            return true;
        }
        return false;
    }

    void update() {
        if (cycleIndex == maxCycle) {
            return;
        }
        if (!propagate(cycleIndex % 10)) {
            return;
        }
        cycleIndex++;
    }
};
