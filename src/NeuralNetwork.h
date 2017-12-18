#pragma once
#include "MatrixManipulation.h"

template<typename T>

/*Three-layer neural network*/
class NeuralNetwork {
private:
	unsigned input_nodes;
	unsigned hidden_nodes;
	unsigned output_nodes;
	T learning_rate;
	MatrixManipulation<T> input_hidden;
	MatrixManipulation<T> hidden_output;
	MatrixManipulation<T> out_hidden;
	MatrixManipulation<T> out_output;

public:

	/*Contructor for a 3-layer neural network
	i_nodes: Number of input nodes
	h_nodes: number of hidden nodes
	o_nodes: number of output nodes*/
	NeuralNetwork(unsigned i_nodes, unsigned h_nodes, unsigned o_nodes, double l_r);
	
	/*Initialize this neural network*/
	void initialize();

	/*Train with an input vector and target vector*/
	void train(const vector<T>& input, const vector<T>& target);

	/*Query. This method returns a vector of numbers indicates the most likely ouput*/
	MatrixManipulation<T> query(const vector<T>& input);

	/*Destructor*/
	~NeuralNetwork();
};
