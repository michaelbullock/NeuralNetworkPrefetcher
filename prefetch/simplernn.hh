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
 */

#ifndef __MEM_CACHE_PREFETCH_SIMPLERNN_HH__
#define __MEM_CACHE_PREFETCH_SIMPLERNN_HH__

// Extends the queued prefetching scheme

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>

#include "mem/cache/prefetch/queued.hh"
#include "params/SimpleRNNPrefetcher.hh"
#include "./Eigen/Dense"

#define MAX_STRIDE_SIZE 10
#define NUM_HIDDENS 64
#define ITERATIONS 100000
#define RECURRENCE_LENGTH 24
#define THETA 0.01

using namespace Eigen;
using namespace std;

class SimpleRNNPrefetcher : public QueuedPrefetcher
{
  //define RNN specific functions and variables here. These are for internal prefetcher use, not for GEM5 visibility
  protected:
  
	struct updates {
		Eigen::Matrix<double, MAX_STRIDE_SIZE, NUM_HIDDENS> synapse_0_update;
		Eigen::Matrix<double, NUM_HIDDENS, MAX_STRIDE_SIZE> synapse_1_update;
		Eigen::Matrix<double, NUM_HIDDENS, NUM_HIDDENS> synapse_h_update;
	};

	std::ofstream addressfile;
	
	//compute sigmoid 
	double sigmoid(double x);

	//compute the derivative of the output of the sigmoid
	double sigmoid_output_to_derivative(double output);

	Matrix<double, 1, MAX_STRIDE_SIZE> sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x);

	Matrix<double, 1, NUM_HIDDENS> sigmoid(Matrix<double, 1, NUM_HIDDENS>& x);

	double dsigmoid(double x);

	Matrix<double, 1, MAX_STRIDE_SIZE> dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x);

	Matrix<double, 1, NUM_HIDDENS> dsigmoid(Matrix<double, 1, NUM_HIDDENS>& x);

	vector<int> create_sequence_count(int start);

	Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> strides_vector_to_onehot(vector<int> strides);

	Matrix<double, 1, MAX_STRIDE_SIZE> argmax(Matrix<double, 1, MAX_STRIDE_SIZE>& x);

	updates grad_clipping(updates grads, double theta);

  //inherited from queued prefetcher
  public:

	SimpleRNNPrefetcher(const SimpleRNNPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
};

#endif // __MEM_CACHE_PREFETCH_SIMPLERNN_HH__
