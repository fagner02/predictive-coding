#include "capped_neuron.hpp"

CappedNeuron::CappedNeuron(bool isInput, NeuronType type) {
    activity = 0;
    prediction = 0;
    this->isInput = isInput;
    this->type = type;
}

void CappedNeuron::reset() {
    activity = 0;
    prediction = 0;
}

void CappedNeuron::recalcWeights() {}

void CappedNeuron::addConnection(capped_neuron_ptr neuron,
                                 ConnectionType type) {
    conn_ptr newConnection =
        std::make_shared<Connection>(Connection{.weight = 0, .type = type});
    this->outgoing.insert(
        {neuron, OutgoingConnection{.connection = newConnection}});
    neuron->incoming.insert({shared_from_this(), newConnection});
}

void CappedNeuron::recalcActivity() {}

void CappedNeuron::update() {}

void CappedNeuron::recalcPrediction() {}

void CappedNeuron::recalcError() { this->error = (activity - prediction); }
