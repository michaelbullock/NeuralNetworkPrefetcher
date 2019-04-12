/*
	Authors: 
		Michael Bullock - bullockm@email.arizona.edu
		Michael Inouye - mikesinouye@email.arizona.edu
		Curt Bansil - curtbansil@email.arizona.edu
		Alli Gilbreath - alligilbreath@email.arizona.edu
	
	ECE 462/562: Computer Architecture and Design
		Course Project
		Dr. Tosiron Adegbija
 */

/**
 * @file
 * Implements a simple RNN prefetcher
   Neural network prefetcher template instantiations
 */

#include "mem/cache/prefetch/simplernn.hh"

#include "base/random.hh"
#include "base/trace.hh"
#include "debug/HWPrefetch.hh"
#include "./Eigen/Dense"

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>

using namespace Eigen;
using namespace std;

SimpleRNNPrefetcher::SimpleRNNPrefetcher(const SimpleRNNPrefetcherParams *p) : QueuedPrefetcher(p) 

{
	
    // Don't consult the prefetcher on instruction accesses
    onInst = false;

}

//compute sigmoid 
double SimpleRNNPrefetcher::sigmoid(double x) {
	return (1 / (1 + exp(-(x))));
}

//compute the derivative of the output of the sigmoid
double SimpleRNNPrefetcher::sigmoid_output_to_derivative(double output) {
	return output * (1 - output);
}

Matrix<double, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()(i) = sigmoid(input.array()(i));
	}

	return input;
}

Matrix<double, 1, NUM_HIDDENS> SimpleRNNPrefetcher::sigmoid(Matrix<double, 1, NUM_HIDDENS>& x)
{
	Matrix<double, 1, NUM_HIDDENS> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()(i) = sigmoid(input.array()(i));
	}

	return input;
}

double SimpleRNNPrefetcher::dsigmoid(double x)
{
	return (sigmoid(x) * (1 - sigmoid(x)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dsigmoid(input.array()[i]);
	}
	return input;
}

Matrix<double, 1, NUM_HIDDENS> SimpleRNNPrefetcher::dsigmoid(Matrix<double, 1, NUM_HIDDENS>& x)
{
	Matrix<double, 1, NUM_HIDDENS> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dsigmoid(input.array()[i]);
	}
	return input;
}

vector<int> SimpleRNNPrefetcher::create_sequence_count(int start) {
	vector<int> x;
	for (int i = start; i < RECURRENCE_LENGTH + 1 + start; i++) {
		x.push_back(i%MAX_STRIDE_SIZE);
	}
	return x;
}

Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::strides_vector_to_onehot(vector<int> strides) {

	Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> onehots;
	onehots.setZero();
	for (unsigned int i = 0; i < strides.size(); i++) {
		Matrix<double, 1, MAX_STRIDE_SIZE> cur_onehot;
		cur_onehot.setZero();
		cur_onehot(0, strides.at(i)) = (double)1;
		onehots.row(i) = cur_onehot;
	}

	return onehots;
}

Matrix<double, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::argmax(Matrix<double, 1, MAX_STRIDE_SIZE>& x) {
	double cur_max = -1;
	int cur_max_index = -1;
	for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
		if (cur_max < x(0, i)) {
			cur_max = x(0, i);
			cur_max_index = i;
		}
	}
	x.setZero();
	x(0, cur_max_index) = 1;
	return x;
}

SimpleRNNPrefetcher::updates SimpleRNNPrefetcher::grad_clipping(updates grads, double theta) {
	double norm = 0;
	norm += grads.synapse_0_update.array().pow(2).sum();
	norm += grads.synapse_1_update.array().pow(2).sum();
	norm += grads.synapse_h_update.array().pow(2).sum();
	norm = sqrt(norm);
	if (norm > theta) {
		grads.synapse_0_update.array() *= (theta / norm);
		grads.synapse_1_update.array() *= (theta / norm);
		grads.synapse_h_update.array() *= (theta / norm);
	}
	return grads;
}

void SimpleRNNPrefetcher::calculatePrefetch(const PacketPtr &pkt, std::vector<AddrPriority> &addresses)
{
    if (!pkt->req->hasPC()) {
        DPRINTF(HWPrefetch, "Ignoring request with no PC.\n");
        return;
    }

    // Get required packet info
    Addr pkt_addr = pkt->getAddr();
    //Addr pc = pkt->req->getPC();
    //bool is_secure = pkt->isSecure();
    //MasterID master_id = useMasterId ? pkt->req->masterId() : 0;
	
	addressfile.open("address_list.txt", std::ios_base::app);
	std::string s = std::to_string(pkt_addr);
	addressfile << s;
	addressfile << "\n";
	//addressfile << std::to_string(pageBytes);
	addressfile.close();
	
	
	
	// ------------RNN LOGIC---------------
	
	// input variables
	double alpha = 10;
	const int input_dim = MAX_STRIDE_SIZE;
	const int hidden_dim = NUM_HIDDENS;
	const int output_dim = MAX_STRIDE_SIZE;

	//initialize rnn weights
	Eigen::Matrix<double, input_dim, hidden_dim> synapse_0 = Eigen::Matrix<double, input_dim, hidden_dim>::Random();
	Eigen::Matrix<double, hidden_dim, output_dim> synapse_1;
	synapse_1.setRandom();
	Eigen::Matrix<double, hidden_dim, hidden_dim> synapse_h;
	synapse_h.setRandom();

	//update 
	Eigen::Matrix<double, input_dim, hidden_dim> synapse_0_update;
	synapse_0_update.setZero();
	Eigen::Matrix<double, hidden_dim, output_dim> synapse_1_update;
	synapse_1_update.setZero();
	Eigen::Matrix<double, hidden_dim, hidden_dim> synapse_h_update;
	synapse_h_update.setZero();


	//training logic
	for (int j = 0; j < ITERATIONS; j++) {
		//generate random start
		int start = std::rand() % (MAX_STRIDE_SIZE);
		//generate sequence
		Eigen::Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs = strides_vector_to_onehot(create_sequence_count(start));


		//place to store our best guess
		Eigen::Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> d;
		d.setZero();

		double overall_error = 0;

		std::vector<Eigen::Matrix<double, 1, output_dim>> layer_2_deltas;
		std::vector<Eigen::Matrix<double, 1, hidden_dim>> layer_1_values;
		layer_1_values.push_back(Eigen::Matrix<double, 1, hidden_dim>::Zero());

		for (int position = 0; position < RECURRENCE_LENGTH; position++) {

			//generate input and output 
			Matrix<double, 1, MAX_STRIDE_SIZE> X = expected_outputs.row(position);
			Matrix<double, 1, MAX_STRIDE_SIZE> y = expected_outputs.row(position + 1);

			// hidden layer
			Eigen::Matrix<double, 1, hidden_dim> layer_1 = X * synapse_0 + layer_1_values.back()*synapse_h;

			layer_1 = sigmoid(layer_1);

			//output layer
			Eigen::Matrix<double, 1, output_dim> layer_2 = layer_1 * synapse_1;

			layer_2 = sigmoid(layer_2);

			// calc error
			Eigen::Matrix<double, 1, output_dim> layer_2_error = y - layer_2;
			layer_2_deltas.push_back((layer_2_error).array() * dsigmoid(layer_2).array());
			overall_error += (double)abs(layer_2_error.sum());

			//estimate
			d.row(position) = argmax(layer_2);

			//store hidden layer so we can print it
			Eigen::Matrix<double, 1, hidden_dim> layer_1_copy = layer_1;
			layer_1_values.push_back(layer_1_copy);
		}

		Eigen::Matrix<double, 1, hidden_dim> future_layer_1_delta = Eigen::Matrix<double, 1, hidden_dim>::Zero();


		//backpropagate
		for (int position = RECURRENCE_LENGTH - 1; position >= 0; position--) {
			Matrix<double, 1, MAX_STRIDE_SIZE> X = expected_outputs.row(position);


			Eigen::Matrix<double, 1, hidden_dim> layer_1 = layer_1_values.at(position + 1);

			Eigen::Matrix<double, 1, hidden_dim> prev_layer_1 = layer_1_values.at(position);

			//error at output layer
			Eigen::Matrix<double, 1, MAX_STRIDE_SIZE> layer_2_delta = layer_2_deltas.at(position);

			//error at hidden layer
			Eigen::Matrix<double, 1, hidden_dim> layer_1_delta = (future_layer_1_delta*synapse_h.transpose() + layer_2_delta * synapse_1.transpose()).array() * dsigmoid(layer_1).array();

			synapse_1_update = synapse_1_update + (layer_1.transpose()*layer_2_delta);
			synapse_h_update = synapse_h_update + (prev_layer_1.transpose()*layer_1_delta);
			synapse_0_update = synapse_0_update + (X.transpose()*layer_1_delta);



			future_layer_1_delta = layer_1_delta;
		}

		updates g;
		g.synapse_0_update = synapse_0_update;
		g.synapse_1_update = synapse_1_update;
		g.synapse_h_update = synapse_h_update;

		g = grad_clipping(g, THETA);

		synapse_0_update = g.synapse_0_update;
		synapse_1_update = g.synapse_1_update;
		synapse_h_update = g.synapse_h_update;

		synapse_0 = synapse_0 + (synapse_0_update * alpha);
		synapse_1 = synapse_1 + (synapse_1_update * alpha);
		synapse_h = synapse_h + (synapse_h_update * alpha);

		synapse_0_update.setZero();
		synapse_1_update.setZero();
		synapse_h_update.setZero();


		std::cout << "error: " << overall_error << std::endl;

		if (j % 100 == 0) {
			
			std::cout.precision(4);
			std::cout << "error: " << overall_error << std::endl;
			std::cout << "Predicted: " << d.row(RECURRENCE_LENGTH-1) << std::endl;
			std::cout << "Actual: " << expected_outputs.row(RECURRENCE_LENGTH) << std::endl;
		}
	}
	
	Addr new_addr = pkt_addr; // This is where the magic happens
	
	// Check to see if the new address to prefetch is within the same page as the current packet address
	if (samePage(pkt_addr, new_addr)) {
		DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
		addresses.push_back(AddrPriority(new_addr, 0));	
	}
	
	else {
		DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
		return;
	}
	
	return;

}

SimpleRNNPrefetcher*
SimpleRNNPrefetcherParams::create()
{
    return new SimpleRNNPrefetcher(this);
}