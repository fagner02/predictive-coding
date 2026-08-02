#include "types.hpp"
#include <map>
#include <memory>
#include <string>

class CappedNeuron;

typedef std::shared_ptr<CappedNeuron> capped_neuron_ptr;

enum ConnectionType { Inhibitory, Excitatory };

struct Connection {
    double weight;
    ConnectionType type;
};

struct OutgoingConnection {
    conn_ptr connection;
};

class CappedNeuron : public std::enable_shared_from_this<CappedNeuron> {
  protected:
    double prediction;
    double error;
    double sd;

  public:
    double activity;
    NeuronType type;
    bool isInput;
    bool isActive = false;
    bool wasActive = false;
    std::string name;
    double lowerBound;
    double upperBound;
    std::map<capped_neuron_ptr, conn_ptr> incoming = {};
    std::map<capped_neuron_ptr, OutgoingConnection> outgoing = {};

    CappedNeuron(bool isClamped = false, NeuronType type = NeuronType::Normal);

    Mat getPrediction();

    Mat getError();

    void reset();

    void receiveInput(double input);

    void recalcWeights();

    void addConnection(capped_neuron_ptr neuron, ConnectionType type);

    void recalcActivity();

    void update();

    void recalcPrediction();

    void recalcError();
};
