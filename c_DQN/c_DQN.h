#ifndef C_DQN_HEADER
#define C_DQN_HEADER


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


struct NetworkState {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double* input_layer;
    double** hidden_layers;
    double* output_layer;
    double* output;

    double** input_weights;
    double*** hidden_weights;
    double** output_weights;
};


struct NeuralNetwork {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double learning_rate;
    double momentum_value;
    int momentum_enabled;

    double** input_weights;
    double*** hidden_weights;
    double** output_weights;

    double** delta_input_weights;
    double*** delta_hidden_weights;
    double** delta_output_weights;
};


struct Event {
    struct NetworkState* network_state;
    int chosen_action;
    double reward;
    struct Event* next_event;
};


struct History {
    int size;
    struct Event* event_head;
};


struct NeuralNetwork* create_network(int input_size, int hidden_amount, int hidden_size, int output_size, double learning_rate, double momentum_value, int momentum_enabled);
void free_neural_network(struct NeuralNetwork* neural_network);

struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, double* const input);
double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size);
void free_network_state(struct NetworkState* network_state);

void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, double* const target_output);
double* create_loss(double* const output, double* const target_output, const int output_size);
double* create_error(double* const layer, const int layer_size, double** const weights, double* const previous_error, const int previous_error_size);
void update_weights(struct NeuralNetwork* neural_network, double* const layer, const int layer_size, double** weights, double** delta_weights, double* const error, const int error_size);

struct History* create_history();
void free_history(struct History* history);
void add_event(struct History* history, struct NetworkState* network_state, int chosen_action, double reward);
void free_event(struct Event* event);
void perform_batch_update_pop_amount(struct NeuralNetwork* neural_network, struct History* history, int pop_amount, const double alpha, const double gamma);
void perform_batch_update_pop_last(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma);
void perform_batch_update_pop_all(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma);
void preform_batch_update_rec(struct NeuralNetwork* neural_network, struct Event* event, short is_first_event, double previous_max_Qvalue, const double alpha, const double gamma);

double sigmoid_function(double x);
double inv_sigmoid_function(double x);
double* create_sigmoid_array(double* const array, const int size);
double* create_inv_sigmoid_array(double* const array, const int size);
double get_max_value(double* const array, const int size);
int get_max_index(double* const array, const int size);

double* create_double_array(const int size, const double initial_value);
double** create_2D_double_array(const int i_size, const int j_size, const double initial_value);
double*** create_3D_double_array(const int i_size, const int j_size, const int k_size, const double initial_value);

double get_random_double(double min, double max);
double* create_double_array_copy(double* const array, const int size);
double** create_2D_double_array_copy(double** const array, const int i_size, const int j_size);
double*** create_3D_double_array_copy(double*** const array, const int i_size, const int j_size, const int k_size);

void free_2D_double_array(double** array, const int size);
void free_3D_double_array(double*** array, const int i_size, const int j_size);

int get_history_size(struct History* history);
int get_input_size(struct NetworkState* const network_state);
int get_hidden_amount(struct NetworkState* const network_state);
int get_hidden_size(struct NetworkState* const network_state);
int get_output_size(struct NetworkState* const network_state);
double* get_input_layer(struct NetworkState* const network_state);
double** get_hidden_layers(struct NetworkState* const network_state);
double* get_output_layer(struct NetworkState* const network_state);
double* get_output(struct NetworkState* const network_state);
int choose_action(struct NetworkState* const network_state);

void display_output(struct NetworkState* const network_state);


#endif