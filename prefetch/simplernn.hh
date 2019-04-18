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
 */

#ifndef __MEM_CACHE_PREFETCH_SIMPLERNN_HH__
#define __MEM_CACHE_PREFETCH_SIMPLERNN_HH__

// Extends the queued prefetching scheme

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>

#define EIGEN_STACK_ALLOCATION_LIMIT 4294967296

#include "mem/cache/prefetch/queued.hh"
#include "params/SimpleRNNPrefetcher.hh"
#include "./Eigen/Dense"


#define MAX_STRIDE_SIZE 512
#define FLUSH_LIMIT 1
#define NUM_HIDDENS 64
#define ITERATIONS 1
#define RECURRENCE_LENGTH 4
#define THETA 0.001
#define ALPHA 100

#define MIN_CONFIDENCE 0.0
#define START_DELAY 500

using namespace Eigen;
using namespace std;

class SimpleRNNPrefetcher : public QueuedPrefetcher
{
  
    //inherited from queued prefetcher
  public:
	struct updates {
		Eigen::Matrix<float, MAX_STRIDE_SIZE, NUM_HIDDENS> synapse_0_update;
		Eigen::Matrix<float, NUM_HIDDENS, MAX_STRIDE_SIZE> synapse_1_update;
		Eigen::Matrix<float, NUM_HIDDENS, NUM_HIDDENS> synapse_h_update;
	};
	
	SimpleRNNPrefetcher(const SimpleRNNPrefetcherParams *p);

    void calculatePrefetch(const PacketPtr &pkt,
                           std::vector<AddrPriority> &addresses);
  
  //define RNN specific functions and variables here. These are for internal prefetcher use, not for GEM5 visibility
  protected:

	float alpha;
	int startCounter;
	int flushCounter;
	
	//initialize rnn weights
	Eigen::Matrix<float, MAX_STRIDE_SIZE, NUM_HIDDENS> synapse_0;
	Eigen::Matrix<float, NUM_HIDDENS, MAX_STRIDE_SIZE> synapse_1;
	Eigen::Matrix<float, NUM_HIDDENS, NUM_HIDDENS> synapse_h;

	//update 
	Eigen::Matrix<float, MAX_STRIDE_SIZE, NUM_HIDDENS> synapse_0_update;
	Eigen::Matrix<float, NUM_HIDDENS, MAX_STRIDE_SIZE> synapse_1_update;
	Eigen::Matrix<float, NUM_HIDDENS, NUM_HIDDENS> synapse_h_update;
	vector<Eigen::Matrix<float, 1, MAX_STRIDE_SIZE>> expected_outputs;
	
	vector<int> addressList;
	vector<int> strideList;
	int strideMap[MAX_STRIDE_SIZE];
	int nextStrideMap[MAX_STRIDE_SIZE];

	std::ofstream addressfile;
	
	//compute sigmoid 
	float sigmoid(float x);

	//compute the derivative of the output of the sigmoid
	float sigmoid_output_to_derivative(float output);

	Matrix<float, 1, MAX_STRIDE_SIZE> sigmoid(Matrix<float, 1, MAX_STRIDE_SIZE>& x);

	Matrix<float, 1, NUM_HIDDENS> sigmoid(Matrix<float, 1, NUM_HIDDENS>& x);

	float dsigmoid(float x);

	Matrix<float, 1, MAX_STRIDE_SIZE> dsigmoid(Matrix<float, 1, MAX_STRIDE_SIZE>& x);

	Matrix<float, 1, NUM_HIDDENS> dsigmoid(Matrix<float, 1, NUM_HIDDENS>& x);

	vector<int> create_sequence_count(int start);

	vector<Matrix<float, 1, MAX_STRIDE_SIZE>> strides_vector_to_onehot(vector<int> strides);
	
	Matrix<float, 1, MAX_STRIDE_SIZE> to_onehot(int stride);

	Matrix<float, 1, MAX_STRIDE_SIZE> argmax(Matrix<float, 1, MAX_STRIDE_SIZE>& x);
	
	float extract_confidence(Matrix<float, 1, MAX_STRIDE_SIZE>& x);

	int extract_stride(Matrix<float, 1, MAX_STRIDE_SIZE>& x);

	updates grad_clipping(updates grads, float theta);

};

#endif // __MEM_CACHE_PREFETCH_SIMPLERNN_HH__
