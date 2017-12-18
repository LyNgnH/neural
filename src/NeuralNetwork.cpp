#pragma once
#include "NeuralNetwork.h"
#include "Math.cpp"
#include "MatrixManipulation.cpp"
#include <iostream>

/*Helpers*/
namespace {

	//Generate random weights
	template<typename T>
	void generate_weights(MatrixManipulation<T>& m, unsigned nodes) {
		T bound = 1 / sqrt(nodes);
		m.fill(-bound, bound);
	}

	//Apply activation function to a matrix
	template<typename T>
	void activate(MatrixManipulation<T>& out) {
		for (unsigned i = 0; i < out.get_num_rows(); i++)
			out.set(i, 0, math_utils::sigmoid(out.get(i, 0)));
	}

	//Compute output given an input vector and weight
	template<typename T>
	void forward_propagate(
		const MatrixManipulation<T>& w, 
		const MatrixManipulation<T>& in, 
		MatrixManipulation<T>& res) 
	{
		res = w + in;
		activate(res);
	}

	/* Helper function to calculate weights with threads*/
	template<typename T>
	void bp_helper(
		const MatrixManipulation<T>& error, 
		const MatrixManipulation<T>& ojt, 
		const MatrixManipulation<T>& ok, 
		const T l_r, MatrixManipulation<T>& weights, 
		unsigned i) {
		for (unsigned j = 0; j < weights.get_num_columns(); j++) {
			T dw = -error.get(i, 0) * ok.get(i, 0) * (1 - ok.get(i, 0)) * ojt.get(0, j);
			T w = weights.get(i, j) - l_r * dw;
			weights.set(i, j, w);
		}
	}

	//Update weight given error, previous and next layers, and the weight matrix
	template<typename T>
	void back_propagate(
		const MatrixManipulation<T>& error, 
		const MatrixManipulation<T>& ojt, 
		const MatrixManipulation<T>& ok, 
		const T l_r, 
		MatrixManipulation<T>& weights)
	{
		vector<thread> threads;
		//Spawn new threads
		for (unsigned i = 0; i < weights.get_num_rows(); i++) {
			threads.push_back(thread(bp_helper<T>, ref(error), ref(ojt), ref(ok), l_r, ref(weights), i));
		}

		//Join threads
		for (thread& thr : threads) {
			thr.join();
		}
	}

	//Wrap a vector inside the matrix manipulation class
	template<typename T>
	MatrixManipulation<T> wrap(const vector<T>& v) {
		matrix<T> r_m(1, v);
		MatrixManipulation<T> ret(r_m);
		return ret.transpose();
	}
}


//Constructor: Initialize general info. and weight matrices
template<typename T>
NeuralNetwork<T>::NeuralNetwork(
	unsigned i_nodes, unsigned h_nodes, 
	unsigned o_nodes, double l_r)
	: 
	input_nodes(i_nodes), hidden_nodes(h_nodes), 
	output_nodes(o_nodes), learning_rate(l_r)
{
}

template<typename T>
void NeuralNetwork<T>::initialize() {
	//Initialize input -> hidden matrix
	input_hidden = MatrixManipulation<T>(hidden_nodes, input_nodes);
	//Initialize hidden -> output matrix
	hidden_output = MatrixManipulation<T>(output_nodes, hidden_nodes);
	//Initialize outputs matrix of hidden layers
	out_hidden = MatrixManipulation<T>(hidden_nodes, 1);
	//Initialize outputs matrix of output layers
	out_output = MatrixManipulation<T>(output_nodes, 1);
	//hidden node = # of links to an input node.
	generate_weights(input_hidden, hidden_nodes);
	//output node = # of links to an hidden node.
	generate_weights(hidden_output, output_nodes);
}

template<typename T>
MatrixManipulation<T> NeuralNetwork<T>::query(const vector<T>& input) {
	MatrixManipulation<T> in = wrap(input);
	forward_propagate(input_hidden, in, out_hidden); //Forward 1 
	forward_propagate(hidden_output, out_hidden, out_output); //Forward 2
	return out_output; //Return output
}

template<typename T>
void NeuralNetwork<T>::train(const vector<T>& data_set, const vector<T>& target) {
	
	//Forward propagate to compute output
	query(data_set);

	//Firt backpropagation to update hidden-output weights
	MatrixManipulation<T> output_error = wrap(target) - out_output;
	MatrixManipulation<T> ojt = out_hidden.transpose();
	back_propagate(output_error, ojt, out_output, learning_rate, hidden_output);

	//Second backpropagation to update input-hidden weights
	MatrixManipulation<T> hidden_error = hidden_output.transpose() + output_error;
	matrix<T> d_m(1, data_set);
	MatrixManipulation<T> oin(d_m);
	back_propagate(hidden_error, oin, out_hidden, learning_rate, input_hidden);
}

template<typename T>
NeuralNetwork<T>::~NeuralNetwork() {

}






