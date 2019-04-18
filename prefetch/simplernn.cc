/*
	Authors: 
		Michael Bullock - bullockm@email.arizona.edu
		Michael Inouye - mikesinouye@email.arizona.edu
	
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
	
	//input variables
	startCounter = 0;
	flushCounter = 0;
	
	synapse_0.setRandom();
	synapse_1.setRandom();
	synapse_h.setRandom();
	synapse_0_update.setZero();
	synapse_1_update.setZero();
	synapse_h_update.setZero();
	
	addressfile.open("address_list.txt");
	addressfile.close();
	
	for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
		strideMap[i] = 0;
		nextStrideMap[i] = 0;
	}

}

//compute sigmoid 
float SimpleRNNPrefetcher::sigmoid(float x) {
	return (1 / (1 + exp(-(x))));
}

//compute the derivative of the output of the sigmoid
float SimpleRNNPrefetcher::sigmoid_output_to_derivative(float output) {
	return output * (1 - output);
}

Matrix<float, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::sigmoid(Matrix<float, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<float, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()(i) = sigmoid(input.array()(i));
	}

	return input;
}


Matrix<float, 1, NUM_HIDDENS> SimpleRNNPrefetcher::sigmoid(Matrix<float, 1, NUM_HIDDENS>& x)
{
	Matrix<float, 1, NUM_HIDDENS> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()(i) = sigmoid(input.array()(i));
	}

	return input;
}


float SimpleRNNPrefetcher::dsigmoid(float x)
{
	return (sigmoid(x) * (1 - sigmoid(x)));
}

Matrix<float, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::dsigmoid(Matrix<float, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<float, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dsigmoid(input.array()[i]);
	}
	return input;
}

Matrix<float, 1, NUM_HIDDENS> SimpleRNNPrefetcher::dsigmoid(Matrix<float, 1, NUM_HIDDENS>& x)
{
	Matrix<float, 1, NUM_HIDDENS> input = x;
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

vector<Matrix<float, 1, MAX_STRIDE_SIZE>> SimpleRNNPrefetcher::strides_vector_to_onehot(vector<int> strides) {

	vector<Matrix<float, 1, MAX_STRIDE_SIZE>> onehots;
	for (unsigned int i = 0; i < strides.size(); i++) {
		Matrix<float, 1, MAX_STRIDE_SIZE> cur_onehot;
		cur_onehot.setZero();
		cur_onehot(0, strides.at(i)) = (float)1;
		onehots.push_back(cur_onehot);
	}
	return onehots;
}
Matrix<float, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::to_onehot(int onehot_index) {
	Matrix<float, 1, MAX_STRIDE_SIZE> cur_onehot;
	cur_onehot.setZero();
	cur_onehot(onehot_index) = (float)1;
	return cur_onehot;
}


Matrix<float, 1, MAX_STRIDE_SIZE> SimpleRNNPrefetcher::argmax(Matrix<float, 1, MAX_STRIDE_SIZE>& x) {
	float cur_max = -1;
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

int SimpleRNNPrefetcher::extract_stride(Matrix<float, 1, MAX_STRIDE_SIZE>& x) {
	float cur_max = -1;
	int cur_max_index = -1;
	for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
		if (cur_max < x(0, i)) {
			cur_max = x(0, i);
			cur_max_index = i;
		}
	}
	return cur_max_index;
}

SimpleRNNPrefetcher::updates SimpleRNNPrefetcher::grad_clipping(SimpleRNNPrefetcher::updates grads, float theta) {
	float norm = 0;
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

float SimpleRNNPrefetcher::extract_confidence(Matrix<float, 1, MAX_STRIDE_SIZE>& x) {
	float cur_max = -1;
	for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
		if (cur_max < x(0, i)) {
			cur_max = x(0, i);
		}
	}
	return cur_max;
}

void SimpleRNNPrefetcher::calculatePrefetch(const PacketPtr &pkt, std::vector<AddrPriority> &addresses)
{
	// Delay start of neural network until startup behavior has finished
	startCounter++;
    if (startCounter < START_DELAY) {
		return;
	}
	
	// Obtain address information
	Addr pkt_addr = pkt->getAddr();
	int address = pkt_addr;
	
	int prediction = pkt_addr;
	float confidence = 0.0;

	// Save a running list of the past MAX_STRIDE_SIZE addresses
	addressList.push_back(address);
	
	// Abort the prediction if there aren't any strides yet
	if (addressList.size() < 2) {
		return;
	}
	// If the prior address list is full, remove the oldest element
	else if (addressList.size() > RECURRENCE_LENGTH) {
		addressList.erase(addressList.begin());
	}
	
	int stride = (addressList.back() - addressList.at(addressList.size() - 2)) / blkSize;

	int onehot_index = -1;
	// Ignore zero strides
	if (stride != 0) {

		// Find stride within table map if it exists or add it
		for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
			if (stride == strideMap[i]) {
				onehot_index = i;
				
				strideList.push_back(stride);
				if (strideList.size() > RECURRENCE_LENGTH + 4) {
					strideList.erase(strideList.begin());
				}
				
				Matrix<float, 1, MAX_STRIDE_SIZE> onehot = to_onehot(onehot_index);
				expected_outputs.push_back(onehot);
				if (expected_outputs.size() > RECURRENCE_LENGTH + 1) {
					expected_outputs.erase(expected_outputs.begin());
				}
				break;
			}
			// strideMap has an empty slot, add new stride to table
			else if (strideMap[i] == 0) {
				onehot_index = i;
				
				strideMap[i] = stride;
				
				addressfile.open("address_list.txt", std::ios_base::app);
				addressfile << "Adding new stride to table: ";
				addressfile << std::to_string(stride);
				addressfile << "\n";
				addressfile.close();	

				strideList.push_back(stride);
				if (strideList.size() > RECURRENCE_LENGTH + 4) {
					strideList.erase(strideList.begin());
				}
				
				Matrix<float, 1, MAX_STRIDE_SIZE> onehot = to_onehot(onehot_index);
				expected_outputs.push_back(onehot);
				if (expected_outputs.size() > RECURRENCE_LENGTH + 1) {
					expected_outputs.erase(expected_outputs.begin());
				}
				break;
			}
		}	
		
		// Fail if it didn't exist and the table has reached MAX_STRIDE_SIZE
		if (onehot_index == -1) {

			// Check to see if the new stride exists within the nextStrideMap
			bool strideExists = false;
			for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
				if (stride == nextStrideMap[i]) {
					strideExists = true;
					break;
				}
			}
			
			// If the current stride isn't in the nextStrideMap, add it, and do flush logic if it's also not in the primary strideMap 
			if (!(strideExists)) {
				
				addressfile.open("address_list.txt", std::ios_base::app);
				addressfile << "Couldn't add a new stride to table map, it's full. This new stride was: ";
				addressfile << std::to_string(stride);
				addressfile << " and the flush counter is ";
				addressfile << std::to_string(flushCounter);
				addressfile << "\n";
				addressfile.close();
				
				nextStrideMap[flushCounter % MAX_STRIDE_SIZE] = stride;
				flushCounter++;
				
				// Check if the target amount of strides couldn't be tabled since the strideMap was full
				if (flushCounter > (FLUSH_LIMIT * MAX_STRIDE_SIZE)) {
					
					addressfile.open("address_list.txt", std::ios_base::app);
					addressfile << "\n\n\n\n\nThe queue is being flushed.\n\n\n\n\n";
					addressfile.close();
					
					flushCounter = 0;
					for (int i = 0; i < MAX_STRIDE_SIZE; i++) {
						strideMap[i] = nextStrideMap[i];
						nextStrideMap[i] = 0;
						synapse_0.setRandom();
						synapse_1.setRandom();
						synapse_h.setRandom();
						addressList.clear();
						strideList.clear();
					}
				}
				
			}

		}
			
	}
	
	if (expected_outputs.empty()) return;
	
	// ------------RNN LOGIC---------------
	
	//training logic
	for (int j = 0; j < ITERATIONS; j++) {

		//place to store our best guess
		Eigen::Matrix<float, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> d;
		d.setZero();

		float overall_error = 0;

		std::vector<Eigen::Matrix<float, 1, MAX_STRIDE_SIZE>> layer_2_deltas;
		std::vector<Eigen::Matrix<float, 1, NUM_HIDDENS>> layer_1_values;
		layer_1_values.push_back(Eigen::Matrix<float, 1, NUM_HIDDENS>::Zero());

		for (int position = 0; position < expected_outputs.size() - 1; position++) {

			//generate input and output 
			Matrix<float, 1, MAX_STRIDE_SIZE> X = expected_outputs.at(position);


			Matrix<float, 1, MAX_STRIDE_SIZE> y = expected_outputs.at(position + 1);

			// hidden layer
			Eigen::Matrix<float, 1, NUM_HIDDENS> layer_1 = X * synapse_0 + layer_1_values.back()*synapse_h;

			layer_1 = sigmoid(layer_1);

			//output layer
			Eigen::Matrix<float, 1, MAX_STRIDE_SIZE> layer_2 = layer_1 * synapse_1;

			layer_2 = sigmoid(layer_2);

			// calc error
			Eigen::Matrix<float, 1, MAX_STRIDE_SIZE> layer_2_error = y - layer_2;
			layer_2_deltas.push_back((layer_2_error).array() * dsigmoid(layer_2).array());
			overall_error += (float)abs(layer_2_error.sum());
			
			//estimate
			d.row(position) = argmax(layer_2);

			//store hidden layer so we can print it
			Eigen::Matrix<float, 1, NUM_HIDDENS> layer_1_copy = layer_1;
			layer_1_values.push_back(layer_1_copy);
		}

		//generate prediction
		Matrix<float, 1, MAX_STRIDE_SIZE> X = expected_outputs.back();
		Eigen::Matrix<float, 1, NUM_HIDDENS> layer_1 = X * synapse_0 + layer_1_values.back()*synapse_h;
		layer_1 = sigmoid(layer_1);
		//output layer
		Eigen::Matrix<float, 1, MAX_STRIDE_SIZE> layer_2 = layer_1 * synapse_1;

		layer_2 = sigmoid(layer_2);
		
		prediction = strideMap[extract_stride(layer_2)];
		
		prediction *= blkSize;
		prediction += address;
		
		confidence = extract_confidence(layer_2);
		
		Eigen::Matrix<float, 1, NUM_HIDDENS> future_layer_1_delta = Eigen::Matrix<float, 1, NUM_HIDDENS>::Zero();

		//backpropagate
		for (int position = expected_outputs.size() - 2; position >= 0; position--) {
			Matrix<float, 1, MAX_STRIDE_SIZE> X = expected_outputs.at(position);


			Eigen::Matrix<float, 1, NUM_HIDDENS> layer_1 = layer_1_values.at(position + 1);

			Eigen::Matrix<float, 1, NUM_HIDDENS> prev_layer_1 = layer_1_values.at(position);

			//error at output layer
			Eigen::Matrix<float, 1, MAX_STRIDE_SIZE> layer_2_delta = layer_2_deltas.at(position);

			//error at hidden layer
			Eigen::Matrix<float, 1, NUM_HIDDENS> layer_1_delta = (future_layer_1_delta*synapse_h.transpose() + layer_2_delta * synapse_1.transpose()).array() * dsigmoid(layer_1).array();

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

		synapse_0 = synapse_0 + (synapse_0_update * ALPHA);
		synapse_1 = synapse_1 + (synapse_1_update * ALPHA);
		synapse_h = synapse_h + (synapse_h_update * ALPHA);

		synapse_0_update.setZero();
		synapse_1_update.setZero();
		synapse_h_update.setZero();

	}
	
	
	// ------------End Neural Network---------------
	
	Addr new_addr = prediction; 
	
	// Ignore negative addresses
	if (prediction < 0) return;
	
	// Ignore zero strides
	if (((prediction - address) / blkSize) == 0) {
		addressfile << "Predicted Stride was 0, ignoring.";
		return;
	}
	
	// Only add address to queue if it is above a certain confidence threshold
	if (confidence > MIN_CONFIDENCE) {
		
		DPRINTF(HWPrefetch, "Queuing prefetch to %#x.\n", new_addr);
		addresses.push_back(AddrPriority(new_addr, 0));
			
		addressfile.open("address_list.txt", std::ios_base::app);
		std::string s = std::to_string(((prediction - address) / blkSize));
		addressfile << "Predicted Stride: ";
		addressfile << s;
		addressfile << " with confidence ";
		addressfile << confidence;
		addressfile << " and prior strides ";
		for (int i = 0; i < strideList.size(); i++) {
			addressfile << std::to_string(strideList.at(i));
			addressfile << " ";
		}
		addressfile << "\n";
		addressfile.close();	
	}
	
	return;

}

SimpleRNNPrefetcher*
SimpleRNNPrefetcherParams::create()
{
    return new SimpleRNNPrefetcher(this);
}
