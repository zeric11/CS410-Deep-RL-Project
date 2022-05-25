#ifndef C_DQN_HEADER
#define C_DQN_HEADER


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>


static const short MAX_THREADS = 8;

static const short BIAS_VALUE = 1;


struct NetworkState {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double* input_layer;
    double** hidden_layers;
    double* output_layer;
};


struct NeuralNetwork {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double learning_rate;
    double momentum_value;
    short momentum_enabled;

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


struct NeuralNetwork* create_neural_network(int input_size, int hidden_amount, int hidden_size, int output_size, double learning_rate, double momentum_value, int momentum_enabled);
void free_neural_network(struct NeuralNetwork* neural_network);

struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, double* const input);
double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size);
void* create_next_layer_thread(void* params_ptr);
void free_network_state(struct NetworkState* network_state);

void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, double* const target_output);
void get_loss(long double* loss, double* const output, double* const target, const int output_size);
void get_error(long double* error, double* const layer, const int layer_size, double** const weights, long double* const previous_error, const int previous_error_size);
void* get_error_thread(void* params_ptr);
void update_weights(struct NeuralNetwork* neural_network, double* const layer, const int layer_size, double** weights, double** delta_weights, long double* const error, const int error_size);
void* update_weights_thread(void* params_ptr);

struct History* create_history();
void free_history(struct History* history);
void add_event(struct History* history, struct NetworkState* network_state, int chosen_action, double reward);
void free_event(struct Event* event);
void perform_batch_update_last(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma);
void perform_batch_update_all(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma);

long double sigmoid_function(long double x);
long double inv_sigmoid_function(long double x);
void apply_sigmoid_to_array(double* dest_array, double* src_array, const int size);
double* create_sigmoid_array(double* const array, const int size);
void apply_inv_sigmoid_to_array(double* dest_array, double* src_array, const int size);
double* create_inv_sigmoid_array(double* const array, const int size);
double get_max_value(double* const array, const int size);
int get_max_index(double* const array, const int size);

double get_random_double(double min, double max);
double* create_double_array(const int size, const int randomize);
double** create_2D_double_array(const int i_size, const int j_size, const int randomize);
double*** create_3D_double_array(const int i_size, const int j_size, const int k_size, const int randomize);
void copy_double_array(double* dest_array, double* const src_array, const int size);
double* create_double_array_copy(double* const array, const int size);
double** create_2D_double_array_copy(double** const array, const int i_size, const int j_size);
double*** create_3D_double_array_copy(double*** const array, const int i_size, const int j_size, const int k_size);
void get_biased_array(double* dest_array, double* const src_array, const int size);
double* create_biased_array_copy(double* const array, const int size);
void free_2D_double_array(double** array, const int size);
void free_3D_double_array(double*** array, const int i_size, const int j_size);
void free_2D_long_double_array(long double** array, const int size);
void free_3D_long_double_array(long double*** array, const int i_size, const int j_size);

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