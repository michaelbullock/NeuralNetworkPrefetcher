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
 * Implements a more complex LSTM RNN prefetcher
 */

#ifndef __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__
#define __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__

// Extends the queued prefetching scheme

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>

#include "mem/cache/prefetch/queued.hh"
#include "params/NeuralNetworkPrefetcher.hh"
#include "./Eigen/Dense"

#define MAX_STRIDE_SIZE 10
#define NUM_HIDDENS 64
#define BATCHES 1
#define RECURRENCE_LENGTH 24

using namespace Eigen;
using namespace std;

class NeuralNetworkPrefetcher : public QueuedPrefetcher
{
  //define neural network specific functions and variables here. These are for internal prefetcher use, not for GEM5 visibility
  protected:
	std::ofstream addressfile;
	
	struct LSTM_FP_outputs {
		Matrix<double, 1, MAX_STRIDE_SIZE>  cell_state_mem;
		Matrix<double, 1, MAX_STRIDE_SIZE> hidden_state;
		Matrix<double, 1, MAX_STRIDE_SIZE> forget_gate;
		Matrix<double, 1, MAX_STRIDE_SIZE> input_gate;
		Matrix<double, 1, MAX_STRIDE_SIZE> cell_state;
		Matrix<double, 1, MAX_STRIDE_SIZE> output_gate;
	};

	struct LSTM_BP_outputs {
		Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> forget_gate_update;
		Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input_gate_update;
		Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> cell_state_update;
		Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> output_gate_update;
		Matrix<double, 1, MAX_STRIDE_SIZE> rnn_cell_state_derivative;
		Matrix<double, 1, MAX_STRIDE_SIZE> rnn_hidden_state_derivative;
	};

	class LSTM {
		private:
			Matrix<double, 1, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> x;
			int xs = 2*MAX_STRIDE_SIZE;
			//expected output (next word)
			Matrix<double, 1, MAX_STRIDE_SIZE> y;
			int ys = MAX_STRIDE_SIZE;
			//cell state memory
			Matrix<double, 1, MAX_STRIDE_SIZE> cell_state_mem;
			//length of recurrent network - number of recurrences, number of words
			int rl = RECURRENCE_LENGTH;
			//learning rate for the network, to be tunable
			double learning_rate;
			//forget gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> forget_gate;
			//input gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input_gate;
			//cell state
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> cell_state;
			//output gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> output_gate;
			//gradients
			//forget gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> grad_forget_gate;
			//input gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> grad_input_gate;
			//cell state
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> grad_cell_state;
			//output gate
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> grad_output_gate;
		
		public:
			LSTM();
			LSTM(double lr);
			void setX(Matrix<double, 1, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& input);
			void setCellStateMem(Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state_mem);
			double sigmoid(double x);
			Matrix<double, 1, MAX_STRIDE_SIZE> sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE> &x);
			double dsigmoid(double x);
			Matrix<double, 1, MAX_STRIDE_SIZE> dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x);
			double htangent(double x);
			Matrix<double, 1, MAX_STRIDE_SIZE> htangent(Matrix<double, 1, MAX_STRIDE_SIZE>& x);
			double dhtangent(double x);
			Matrix<double, 1, MAX_STRIDE_SIZE> dhtangent(Matrix<double, 1, MAX_STRIDE_SIZE>& x);
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> square(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& x);
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> root(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& x);
			LSTM_FP_outputs forwardProp();
			LSTM_BP_outputs backwardProp(Matrix<double, 1, MAX_STRIDE_SIZE>& error,
				Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state_mem,
				Matrix<double, 1, MAX_STRIDE_SIZE>& forget_gate,
				Matrix<double, 1, MAX_STRIDE_SIZE>& input_gate,
				Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state,
				Matrix<double, 1, MAX_STRIDE_SIZE>& output_gate,
				Matrix<double, 1, MAX_STRIDE_SIZE>& rnn_cell_state_update,
				Matrix<double, 1, MAX_STRIDE_SIZE>& rnn_hidden_state_update);
			void update(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& forget_gate_update,
				Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& input_gate_update,
				Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& cell_state_update,
				Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& output_gate_update);
	};

	class RecurrentNeuralNetwork {
		private:
			//initial input (first word)
			Matrix<double, 1, MAX_STRIDE_SIZE> x;
			int xs = MAX_STRIDE_SIZE;
			//expected output (next word)
			Matrix<double, 1, MAX_STRIDE_SIZE> y;
			int ys = MAX_STRIDE_SIZE;
			//weight matrix for interpreting lstm results
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> w;
			//matrix for RMS propagation
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> G;
			//length of recurrent network - number of recurrences, number of words
			int rl = RECURRENCE_LENGTH;
			//learning rate for the network, to be tunable
			double learning_rate;
			//matrix for storing inputs
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> inputs;
			//matrix for storing cell states
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> cell_states;
			//Matrix for storing outputs
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> outputs;
			//matrix for storing hidden states
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> hidden_states;
			//forget gate
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> forget_gate;
			//input gate
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> input_gate;
			//cell state
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> cell_state;
			//output gate
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> output_gate;
			//Matrix for expected outputs
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs;
			//lstm cell w. (input, output, amount of recurrence, and learning rate as its constructor
			LSTM LSTM_cell;
		public:
			RecurrentNeuralNetwork();
			RecurrentNeuralNetwork(double learning_rate, Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs);
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> square(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE>& x);
			Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> root(Matrix<double, MAX_STRIDE_SIZE,  MAX_STRIDE_SIZE>& x);
			//activation function
			double sigmoid(double x); 
			Matrix<double, 1, MAX_STRIDE_SIZE> sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE> &x);
			double dsigmoid(double x);
			Matrix<double, 1, MAX_STRIDE_SIZE> dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x);
			Matrix<double, 1, MAX_STRIDE_SIZE> argmax(Matrix<double, 1, MAX_STRIDE_SIZE>& x);
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> forwardProp();
			double backProp();
			void update(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE>& weight);
			Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> sample();
	};
		
	class Driver {
		public:
			RecurrentNeuralNetwork RNN;
			Driver();
			Driver(double learning_rate, Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs);
			void train_and_generate();
	};
	
	vector<int> create_sequence_count();

	Matrix<double, RECURRENCE_LENGTH+1, MAX_STRIDE_SIZE> strides_vector_to_onehot(vector<int> strides);
	
	

  //inherited from queued prefetcher
  public:

    NeuralNetworkPrefetcher(const NeuralNetworkPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
};

#endif // __MEM_CACHE_PREFETCH_NEURALNETWORK_HH__
