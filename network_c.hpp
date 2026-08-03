#include "capped_neuron.hpp"
#include "types.hpp"
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

class NetworkC {
    std::vector<capped_neuron_ptr> neurons;
    std::vector<capped_neuron_ptr> input;
    std::vector<capped_neuron_ptr> output;

    std::set<capped_neuron_ptr> nextParents = {};

    size_t cycleIndex = 0;

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

    void propagate(double data) {
        std::set<capped_neuron_ptr> parents = {input.at(0)};
        std::set<capped_neuron_ptr> visited = {};
        input.at(0)->activity = data;

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

                double center = child->lowerBound +
                                (child->upperBound - child->lowerBound) / 2;

                double increase = a + n->activity * (w + 0.1);
                double decrease = a + n->activity * (w - 0.1);

                // Remember to add breaks for when the activity goes off the
                // bounds
                if (abs(increase - center) < abs(decrease - center)) {
                    outCon.connection->weight = w + 0.1;
                } else {
                    outCon.connection->weight = w - 0.1;
                }
            }
        }
    }

    void update() {
        propagate(cycleIndex % 10);
        cycleIndex++;
    }
};
