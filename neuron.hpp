#pragma once
#include "types.hpp"
#include <map>
#include <memory>

struct Connection {
    Mat activityWeights;
};
struct OutgoingConnection {
    conn_ptr incoming;
};

Mat sigmoidDerivative(Mat m);

Mat sigmoid(Mat m);
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
           NeuronType type = NeuronType::Normal);

    virtual Mat getActivity();

    virtual Mat getPrediction();

    virtual Mat getError();

    virtual void reset();

    virtual void setActivity(Mat activity);

    virtual void recalcWeights();

    virtual void addConnection(neuron_ptr neuron);

    virtual void recalcActivity();

    virtual void update();

    virtual void recalcPrediction();

    virtual void recalcError();
};
