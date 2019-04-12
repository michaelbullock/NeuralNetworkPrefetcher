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
 * Implements a more complex RNN Prefetcher
   Neural network prefetcher template instantiations
 */

#include "mem/cache/prefetch/neuralnetwork.hh"

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

NeuralNetworkPrefetcher::NeuralNetworkPrefetcher(const NeuralNetworkPrefetcherParams *p) : QueuedPrefetcher(p) 

{
	
    // Don't consult the prefetcher on instruction accesses
    onInst = false;

}

NeuralNetworkPrefetcher::LSTM::LSTM() {
	
}

NeuralNetworkPrefetcher::LSTM::LSTM(double lr) {
	x.setZero();
	y.setZero();
	cell_state_mem.setZero();
	learning_rate = lr;
	forget_gate.setRandom();
	input_gate.setRandom();
	cell_state.setRandom();
	output_gate.setRandom();
	grad_forget_gate.setZero();
	grad_input_gate.setZero();
	grad_cell_state.setZero();
	grad_output_gate.setZero();
}

void NeuralNetworkPrefetcher::LSTM::setX(Matrix<double, 1, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& input){
	this->x = input;
}

void NeuralNetworkPrefetcher::LSTM::setCellStateMem(Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state_mem) {
	this->cell_state_mem = cell_state_mem;
}

double NeuralNetworkPrefetcher::LSTM::sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{	
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = sigmoid(input.array()[i]);
	}

	return input;
}

double NeuralNetworkPrefetcher::LSTM::dsigmoid(double x)
{
	return (this->sigmoid(x) * (1 - this->sigmoid(x)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dsigmoid(input.array()[i]);
	}
	return input;
}

double NeuralNetworkPrefetcher::LSTM::htangent(double x)
{
	return (tanh(x));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::htangent(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = htangent(input.array()[i]);
	}
	return input;
}

double NeuralNetworkPrefetcher::LSTM::dhtangent(double x)
{
	return (1-(pow(tanh(x),2)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::dhtangent(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dhtangent(input.array()[i]);
	}
	return input;
}

Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::square(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& x) {
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.array().size(); i++) {
		input.array()(i) = pow(input.array()(i),2);
	}
	return input;
}

Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::LSTM::root(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& x)
{
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.array().size(); i++) {
		input.array()(i) = sqrt((input.array()(i), 2));
	}
	return input;
}


NeuralNetworkPrefetcher::LSTM_FP_outputs NeuralNetworkPrefetcher::LSTM::forwardProp() {
	Matrix<double, 1, MAX_STRIDE_SIZE> new_forget_gate = x*forget_gate.transpose();
	new_forget_gate = sigmoid(new_forget_gate);
	cell_state_mem = cell_state_mem.array() * new_forget_gate.array();
	Matrix<double, 1, MAX_STRIDE_SIZE> new_input_gate = (x * input_gate.transpose());
	new_input_gate = sigmoid(new_input_gate);
	Matrix<double, 1, MAX_STRIDE_SIZE> new_cell_state = (x * cell_state.transpose());
	new_cell_state = htangent(new_cell_state);
	//update internal cell state
	cell_state_mem.array() = cell_state_mem.array() + (new_input_gate.array() * new_cell_state.array());
	Matrix<double, 1, MAX_STRIDE_SIZE> new_output_gate = (x * output_gate.transpose());
	new_output_gate = sigmoid(new_output_gate);
	//update y 
	y = new_output_gate.array() * htangent(cell_state_mem).array();
	LSTM_FP_outputs fp;
	fp.cell_state_mem = cell_state_mem;
	fp.hidden_state = y;
	fp.forget_gate = new_forget_gate;
	fp.input_gate = new_input_gate;
	fp.cell_state = new_cell_state;
	fp.output_gate = new_output_gate;
	return fp;
}

//fix pass by reference here if need be
NeuralNetworkPrefetcher::LSTM_BP_outputs NeuralNetworkPrefetcher::LSTM::backwardProp(
	Matrix<double, 1, MAX_STRIDE_SIZE>& error, 
	Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state_mem, 
	Matrix<double, 1, MAX_STRIDE_SIZE>& forget_gate,
	Matrix<double, 1, MAX_STRIDE_SIZE>& input_gate,
	Matrix<double, 1, MAX_STRIDE_SIZE>& cell_state,
	Matrix<double, 1, MAX_STRIDE_SIZE>& output_gate,
	Matrix<double, 1, MAX_STRIDE_SIZE>& rnn_cell_state_update,
	Matrix<double, 1, MAX_STRIDE_SIZE>& rnn_hidden_state_update) {
	//error = hidden state + previous error, clip this val between -6 and 6
	error = (error.array() + rnn_hidden_state_update.array());
	error = error.array().min(6);
	error = error.array().max(-6);

	//multiply error by the activated cell state to compute the output derivative
	Matrix<double, 1, MAX_STRIDE_SIZE> output_derivative = htangent(this->cell_state_mem).array() * error.array();

	//compute output update
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> output_update = (output_derivative.array()*dsigmoid(output_gate).array()).matrix().transpose() * this->x;

	//compute cell state derivative
	Matrix<double, 1, MAX_STRIDE_SIZE> cell_state_derivative = (error.array() * output_gate.array() * dhtangent(this->cell_state_mem).array()) + rnn_cell_state_update.array();
	cell_state_derivative = cell_state_derivative.array().min(6);
	cell_state_derivative = cell_state_derivative.array().max(-6);
	
	//compute derivitave of cell = deriv of cell state * input
	Matrix<double, 1, MAX_STRIDE_SIZE> cell_derivative = cell_state_derivative.array() * input_gate.array();

	//cell update = (deriv cell coeffwise* activated cell) * input
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> cell_state_update = (cell_derivative.array() * dhtangent(cell_state).array()).matrix().transpose() * this->x;

	//derivative of the input 
	Matrix<double, 1, MAX_STRIDE_SIZE> input_derivative = cell_state_derivative.array() * cell_state.array();

	//input update
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input_update = (input_derivative.array() * dsigmoid(input_gate).array()).matrix().transpose() * this->x;

	//derivative of forget
	Matrix<double, 1, MAX_STRIDE_SIZE> forget_derivative = cell_state_derivative.array() * cell_state_mem.array();

	//forget update
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> forget_update = (forget_derivative.array() * dsigmoid(forget_gate).array()).matrix().transpose() * this->x;

	//derivative of cell state output
	Matrix<double, 1, MAX_STRIDE_SIZE> rnn_cell_state_derivative = (cell_state_derivative.array() * forget_gate.array());

	//derivative of the hidden state
	Matrix<double, 1, MAX_STRIDE_SIZE> rnn_hidden_state_derivative = (cell_derivative*this->cell_state).block(0, 0, 1, MAX_STRIDE_SIZE).array() + (output_derivative*this->output_gate).block(0, 0, 1, MAX_STRIDE_SIZE).array() + (input_derivative*this->input_gate).block(0, 0, 1, MAX_STRIDE_SIZE).array() + (forget_derivative*this->forget_gate).block(0, 0, 1, MAX_STRIDE_SIZE).array();

	LSTM_BP_outputs bp;
	bp.cell_state_update = cell_state_update;
	bp.forget_gate_update = forget_update;
	bp.input_gate_update = input_update;
	bp.output_gate_update = output_update;
	bp.rnn_cell_state_derivative = rnn_cell_state_derivative;
	bp.rnn_hidden_state_derivative = rnn_hidden_state_derivative;

	return bp;
}

void NeuralNetworkPrefetcher::LSTM::update(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& forget_gate_update, 
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& input_gate_update, 
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& cell_state_update, 
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE>& output_gate_update) {
	//update gradients
	this->grad_forget_gate = (0.9 * this->grad_forget_gate).array() + (0.1 * square(forget_gate_update)).array();
	this->grad_input_gate = (0.9 * this->grad_input_gate).array() + (0.1 * square(input_gate_update)).array();
	this->grad_cell_state = (0.9 * this->grad_cell_state).array() + (0.1 * square(cell_state_update)).array();
	this->grad_output_gate = (0.9 * this->grad_output_gate).array() + (0.1 * square(output_gate_update)).array();

	//update gates using gradients
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> forget = (0.00000001 + this->grad_forget_gate.array());
	this->forget_gate = this->forget_gate.array() - (((this->learning_rate) / (root(forget).array())) * forget_gate_update.array());

	//Commented out TODO FIX: Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> input = (0.00000001 + this->grad_input_gate.array());
	this->input_gate = this->input_gate.array() - (((this->learning_rate) / (root(grad_input_gate).array())) * input_gate_update.array());
	
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> cell = (0.00000001 + this->grad_cell_state.array());
	this->cell_state = this->cell_state.array() - (((this->learning_rate) / (root(cell).array())) * cell_state_update.array());


	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> output = (0.00000001 + this->grad_output_gate.array());
	this->output_gate = this->output_gate.array() - (((this->learning_rate) / (root(output).array())) * output_gate_update.array());

	return;
}






NeuralNetworkPrefetcher::RecurrentNeuralNetwork::RecurrentNeuralNetwork()
{
}

//init the network
NeuralNetworkPrefetcher::RecurrentNeuralNetwork::RecurrentNeuralNetwork(double lr, Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> eo) {
	x = Matrix<double, 1, MAX_STRIDE_SIZE>::Zero();
	y.setZero();
	w.setRandom();
	G.setZero();
	learning_rate = lr;
	inputs.setZero();
	cell_states.setZero();
	outputs.setZero();
	hidden_states.setZero();
	forget_gate.setZero();
	input_gate.setZero();
	cell_state.setZero();
	output_gate.setZero();
	expected_outputs = eo;
	LSTM_cell = LSTM(lr);
}

Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::square(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.array().size(); i++) {
		input.array()(i) = pow(input.array()(i), 2);
	}
	return input;
}

Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::root(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.array().size(); i++) {
		input.array()(i) = sqrt((input.array()(i), 2));
	}
	return input;
}

double NeuralNetworkPrefetcher::RecurrentNeuralNetwork::sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::sigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()(i) = sigmoid(input.array()(i));
	}

	return input;
}

double NeuralNetworkPrefetcher::RecurrentNeuralNetwork::dsigmoid(double x)
{
	return (this->sigmoid(x) * (1 - this->sigmoid(x)));
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::dsigmoid(Matrix<double, 1, MAX_STRIDE_SIZE>& x)
{
	Matrix<double, 1, MAX_STRIDE_SIZE> input = x;
	for (int i = 0; i < input.size(); i++) {
		input.array()[i] = dsigmoid(input.array()[i]);
	}
	return input;
}

Matrix<double, 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::argmax(Matrix<double, 1, MAX_STRIDE_SIZE>& x) {
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

Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::forwardProp() {
	for (int i = 1; i < RECURRENCE_LENGTH + 1; i++) {
		Matrix<double, 1, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> new_x;
		new_x << hidden_states.row(i - 1), this->x;
		this->LSTM_cell.setX(new_x);
		LSTM_FP_outputs fp = LSTM_cell.forwardProp();
		this->cell_states.row(i) = fp.cell_state_mem;
		this->hidden_states.row(i) = fp.hidden_state;
		this->forget_gate.row(i) = fp.forget_gate;
		this->input_gate.row(i) = fp.input_gate;
		this->cell_state.row(i) = fp.cell_state;
		this->output_gate.row(i) = fp.output_gate;

		Matrix<double, 1, MAX_STRIDE_SIZE> between = fp.hidden_state * this->w;
		this->outputs.row(i) = sigmoid(between);
		this->x = this->expected_outputs.row(i - 1);
	}

	return this->outputs;
}

double NeuralNetworkPrefetcher::RecurrentNeuralNetwork::backProp() {
	double total_error = 0;
	
	//init matrices for gradient updates
	//rnn level gradients
	//cell state
	Matrix<double, 1, MAX_STRIDE_SIZE> derivative_cell_state;
	derivative_cell_state.setZero();

	//hidden state
	Matrix<double, 1, MAX_STRIDE_SIZE> derivative_hidden_state;
	derivative_hidden_state.setZero();

	//weight matrix
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> total_weight_update;
	total_weight_update.setZero();

	//lstm level gradients
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> total_forget_gate_update;
	total_forget_gate_update.setZero();
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> total_input_gate_update;
	total_input_gate_update.setZero();
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> total_cell_state_update;
	total_cell_state_update.setZero();
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> total_output_gate_update;
	total_output_gate_update.setZero();
	
	//backpropagation through recurrences
	for (int i = RECURRENCE_LENGTH; i > 0; i--) {
		//error is the calculated output stored in outputs minus the expected output stored in expected_outputs at a given timestep
		Matrix<double, 1, MAX_STRIDE_SIZE> error = this->outputs.row(i).array() - this->expected_outputs.row(i).array();

		//now we calculate update val for weight matrix
		Matrix<double, 1, MAX_STRIDE_SIZE> cur_output = outputs.row(i);
		total_weight_update = total_weight_update.array() + (hidden_states.row(i).transpose() * (error.array() * dsigmoid(cur_output).array()).matrix()).array();

		//propagate error back
	
		error = (error * this->w).array() /* dsigmoid(cur_output).array()*/;

		//init lstm x
		Matrix<double, 1, 2 * MAX_STRIDE_SIZE> new_x;
		new_x << this->hidden_states.row(i - 1), this->inputs.row(i);
		LSTM_cell.setX(new_x);

		//init lstm cell state
		Matrix<double, 1, MAX_STRIDE_SIZE> new_cell_state_mem = this->cell_state.row(i);
		LSTM_cell.setCellStateMem(new_cell_state_mem);

		
		Matrix<double, 1, MAX_STRIDE_SIZE> lstm_bp_cells = this->cell_states.row(i - 1);
		Matrix<double, 1, MAX_STRIDE_SIZE> lstm_bp_forget = this->forget_gate.row(i);
		Matrix<double, 1, MAX_STRIDE_SIZE> lstm_bp_input = this->input_gate.row(i);
		Matrix<double, 1, MAX_STRIDE_SIZE> lstm_bp_cell = this->cell_state.row(i);
		Matrix<double, 1, MAX_STRIDE_SIZE> lstm_bp_output = this->output_gate.row(i);

		LSTM_BP_outputs bp = LSTM_cell.backwardProp(error, lstm_bp_cells, lstm_bp_forget, lstm_bp_input, lstm_bp_cell, lstm_bp_output, derivative_cell_state, derivative_hidden_state);
		derivative_cell_state = bp.rnn_cell_state_derivative;
		derivative_hidden_state = bp.rnn_hidden_state_derivative;
		total_error = error.sum();

		//store all gradient updates
		total_forget_gate_update = total_forget_gate_update.array() + bp.forget_gate_update.array();
		total_input_gate_update = total_input_gate_update.array() + bp.input_gate_update.array();
		total_cell_state_update = total_cell_state_update.array() + bp.cell_state_update.array();
		total_output_gate_update = total_output_gate_update.array() + bp.output_gate_update.array();
		
	}
	
	Matrix<double, MAX_STRIDE_SIZE, 2 * MAX_STRIDE_SIZE> forget_gate_update_by_recurrence = total_forget_gate_update.array() / RECURRENCE_LENGTH;
	Matrix<double, MAX_STRIDE_SIZE, 2 * MAX_STRIDE_SIZE> input_gate_update_by_recurrence = total_input_gate_update.array() / RECURRENCE_LENGTH;
	Matrix<double, MAX_STRIDE_SIZE, 2 * MAX_STRIDE_SIZE> cell_state_update_by_recurrence = total_cell_state_update.array() / RECURRENCE_LENGTH;
	Matrix<double, MAX_STRIDE_SIZE, 2 * MAX_STRIDE_SIZE> output_gate_update_by_recurrence = total_output_gate_update.array() / RECURRENCE_LENGTH;

	//update lstm matrices with accumulated gradient updates
	this->LSTM_cell.update(forget_gate_update_by_recurrence, input_gate_update_by_recurrence, cell_state_update_by_recurrence, output_gate_update_by_recurrence);

	update(total_weight_update);

	return total_error;
}

void NeuralNetworkPrefetcher::RecurrentNeuralNetwork::update(Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE>& weight) {
	//rmsprop
	this->G = 0.9 * this->G.array() + (0.1 * square(weight).array());
	Matrix<double, MAX_STRIDE_SIZE, MAX_STRIDE_SIZE> inner_root = (this->G.array() + 0.00000001);
	this->w = this->w.array() - ((this->learning_rate / (root(inner_root).array())) * weight.array());
	return;
}

Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::RecurrentNeuralNetwork::sample() {

	for (int i = 1; i < RECURRENCE_LENGTH + 1; i++) {
		Matrix<double, 1, MAX_STRIDE_SIZE + MAX_STRIDE_SIZE> new_x;
		new_x << hidden_states.row(i - 1), this->x;
		this->LSTM_cell.setX(new_x);
		LSTM_FP_outputs fp = this->LSTM_cell.forwardProp();
		this->x = argmax(this->x);
		this->inputs.row(i) = this->x;
		this->cell_states.row(i) = fp.cell_state_mem;
		this->hidden_states.row(i) = fp.hidden_state;
		this->forget_gate.row(i) = fp.forget_gate;
		this->input_gate.row(i) = fp.input_gate;
		this->cell_state.row(i) = fp.cell_state;
		this->output_gate.row(i) = fp.output_gate;

		Matrix<double, 1, MAX_STRIDE_SIZE> sig_input = fp.hidden_state * this->w;
		this->outputs.row(i) = sigmoid(sig_input);
		Matrix<double, 1, MAX_STRIDE_SIZE> out = this->outputs.row(i);
		this->outputs.row(i) = argmax(out);
		Matrix<double, 1, MAX_STRIDE_SIZE> new_new_x = this->outputs.row(i);
		this->x = argmax(new_new_x);
	}
	return this->outputs;
}




NeuralNetworkPrefetcher::Driver::Driver()
{
}

NeuralNetworkPrefetcher::Driver::Driver(double learning_rate, Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs) {
	this->RNN = RecurrentNeuralNetwork(learning_rate, expected_outputs);
}

void NeuralNetworkPrefetcher::Driver::train_and_generate() {
	for (int i = 0; i < 20000; i++) {
		RNN.forwardProp();
		
		/* Commented out TODO FIX:
		double error = RNN.backProp();
		
		
		cout << "Error on iteration " << i << ": " << error << endl;
		if ((i % 500) == 0) {
			
			Matrix<double, RECURRENCE_LENGTH +1, MAX_STRIDE_SIZE> output = RNN.sample();
			Matrix<double, 1, MAX_STRIDE_SIZE> out = output.row(1);
			cout << "here " << endl;

		}*/
	}
}


vector<int> NeuralNetworkPrefetcher::create_sequence_count() {
	vector<int> x;
	for (int i = 0; i < RECURRENCE_LENGTH + 1; i++) {
		x.push_back(i%MAX_STRIDE_SIZE);
	}
	return x;
}

Matrix<double, RECURRENCE_LENGTH+1, MAX_STRIDE_SIZE> NeuralNetworkPrefetcher::strides_vector_to_onehot(vector<int> strides) {

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


void NeuralNetworkPrefetcher::calculatePrefetch(const PacketPtr &pkt, std::vector<AddrPriority> &addresses)
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
	
	
	/* Insert nn code here */
	
	Matrix<double, RECURRENCE_LENGTH + 1, MAX_STRIDE_SIZE> expected_outputs = strides_vector_to_onehot(create_sequence_count());
	double lr = 0.005;
	Driver d(lr, expected_outputs);

	d.train_and_generate();
	
	
	
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

NeuralNetworkPrefetcher*
NeuralNetworkPrefetcherParams::create()
{
    return new NeuralNetworkPrefetcher(this);
}