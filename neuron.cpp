#include "neuron.hpp"

Neuron::Neuron(size_t rows, size_t cols, bool isInput, NeuronType type) {
    activity = Mat::Random(rows, cols) * 0.1;
    prediction = Mat::Random(rows, cols) * 0.1;
    error = Mat::Ones(rows, cols);
    this->isInput = isInput;
    this->type = type;
}

Mat Neuron::getActivity() { return this->activity; }

Mat Neuron::getPrediction() { return this->prediction; }

Mat Neuron::getError() { return this->error; }

void Neuron::reset() {
    size_t rows = activity.rows();
    size_t cols = activity.cols();

    activity = Mat::Random(rows, cols) * 0.1;
    prediction = Mat::Random(rows, cols) * 0.1;
}

void Neuron::setActivity(Mat activity) { this->activity = activity; }

void Neuron::recalcWeights() {
    for (auto &[neuron, connection] : incoming) {
        connection->activityWeights +=
            this->error.cwiseProduct(
                (this->sd).cwiseProduct(neuron->activity)) *
            2;
    }
}

void Neuron::addConnection(neuron_ptr neuron) {
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

void Neuron::recalcActivity() {
    if (isInput) {
        return;
    }
    Mat delta = -error;

    std::vector<neuron_ptr> toErase = {};

    for (const auto &[neuron, connection] : outgoing) {
        const auto thisWeights = connection.incoming->activityWeights;

        delta +=
            thisWeights.cwiseProduct(neuron->error.cwiseProduct(neuron->sd));
    }
    this->activity += delta * 0.1;
}

void Neuron::update() {
    recalcPrediction();
    recalcError();
    recalcActivity();
}

void Neuron::recalcPrediction() {
    this->prediction.setZero();
    for (const auto &[neuron, connection] : incoming) {
        this->prediction +=
            neuron->activity.cwiseProduct(connection->activityWeights);
    }
    sd = sigmoidDerivative(this->prediction);
    this->prediction = sigmoid(this->prediction);
}

void Neuron::recalcError() { this->error = (activity - prediction); }
