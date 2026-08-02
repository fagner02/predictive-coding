#pragma once
#include <Eigen/Dense>

typedef Eigen::MatrixXd Mat;

enum NeuronType { Output, Input, Normal, Bias };

struct Connection;
class Neuron;

typedef std::shared_ptr<Neuron> neuron_ptr;
typedef std::shared_ptr<Connection> conn_ptr;
