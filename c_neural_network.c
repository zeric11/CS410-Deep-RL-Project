// Compile to use with Python:  gcc -Ofast -fPIC -shared -o c_neural_network.so c_neural_network.c
// Compile for testing:         gcc -Ofast -g c_neural_network.c -o c_neural_network -lm
// Check for mem-leaks:         valgrind --tool=memcheck --leak-check=yes -s ./c_neural_network


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


struct NeuralNetwork* create_network(int input_size, int hidden_amount, int hidden_size, int output_size, double learning_rate, double momentum_value, int momentum_enabled) {
    struct NeuralNetwork* neural_network = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));

    neural_network->input_size = input_size;
    neural_network->hidden_amount = hidden_amount;
    neural_network->hidden_size = hidden_size;
    neural_network->output_size = output_size;

    neural_network->learning_rate = learning_rate;
    neural_network->momentum_value = momentum_value;
    neural_network->momentum_enabled = momentum_enabled;

    srand((int)time(NULL));

    // Allocating weights.
    neural_network->input_weights = create_2D_double_array(input_size, hidden_size, 0);
    neural_network->hidden_weights = create_3D_double_array(hidden_amount - 1, hidden_size, hidden_size, 0);
    neural_network->output_weights = create_2D_double_array(hidden_size, output_size, 0);

    // Allocating delta weights.
    neural_network->delta_input_weights = create_2D_double_array(input_size, hidden_size, 0);
    neural_network->delta_hidden_weights = create_3D_double_array(hidden_amount - 1, hidden_size, hidden_size, 0);
    neural_network->delta_output_weights = create_2D_double_array(hidden_size, output_size, 0);

    return neural_network;
}


void free_neural_network(struct NeuralNetwork* neural_network) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    if(neural_network) {
        if(neural_network->input_weights) {
            free_2D_double_array(neural_network->input_weights, input_size);
        }
        if(neural_network->hidden_weights) {
            free_3D_double_array(neural_network->hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(neural_network->output_weights) {
            free_2D_double_array(neural_network->output_weights, hidden_size);
        }
        if(neural_network->delta_input_weights) {
            free_2D_double_array(neural_network->delta_input_weights, input_size);
        }
        if(neural_network->delta_hidden_weights) {
            free_3D_double_array(neural_network->delta_hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(neural_network->delta_output_weights) {
            free_2D_double_array(neural_network->delta_output_weights, hidden_size);
        }
        free(neural_network);
    }
    neural_network = NULL;
}


void free_network_state(struct NetworkState* network_state) {
    const int input_size = network_state->input_size;
    const int hidden_amount = network_state->hidden_amount;
    const int hidden_size = network_state->hidden_size;
    const int output_size = network_state->output_size;

    if(network_state) {
        if(network_state->input_layer) {
            free(network_state->input_layer);
        }
        if(network_state->hidden_layers) {
            free_2D_double_array(network_state->hidden_layers, hidden_amount);
        }
        if(network_state->output_layer) {
            free(network_state->output_layer);
        }
        if(network_state->output) {
            free(network_state->output);
        }
        if(network_state->input_weights) {
            free_2D_double_array(network_state->input_weights, input_size);
        }
        if(network_state->hidden_weights) {
            free_3D_double_array(network_state->hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(network_state->output_weights) {
            free_2D_double_array(network_state->output_weights, hidden_size);
        }
        free(network_state);
    }
    network_state = NULL;
}


struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, double* const input) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    struct NetworkState* network_state = (struct NetworkState*)malloc(sizeof(struct NetworkState));

    network_state->input_size = input_size;
    network_state->hidden_amount = hidden_amount;
    network_state->hidden_size = hidden_size;
    network_state->output_size = output_size;

    // As the network propagates through each layer, the layer that results from each step is stored in the network_state.
    network_state->input_layer = create_double_array_copy(input, input_size);
    network_state->hidden_layers = (double**)malloc(hidden_amount * sizeof(double*));
    network_state->hidden_layers[0] = create_next_layer(input, input_size, neural_network->input_weights, hidden_size);
    for(register int i = 0; i < hidden_amount - 1; ++i) {
        network_state->hidden_layers[i + 1] = create_next_layer(network_state->hidden_layers[i], hidden_size, neural_network->hidden_weights[i], hidden_size);
    }
    network_state->output_layer = create_next_layer(network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, output_size);
    network_state->output = create_inv_sigmoid_array(network_state->output_layer, output_size);

    network_state->input_weights = create_2D_double_array_copy(neural_network->input_weights, input_size, hidden_size);
    network_state->hidden_weights = create_3D_double_array_copy(neural_network->hidden_weights, hidden_amount - 1, hidden_size, hidden_size);
    network_state->output_weights = create_2D_double_array_copy(neural_network->output_weights, hidden_size, output_size);
    
    return network_state;
}


double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size) {
    double* next_layer = (double*)malloc(next_layer_size * sizeof(double));
    for(register int i = 0; i < next_layer_size; ++i) {
        long double total = 0;
        for(register int j = 0; j < layer_size; ++j) {
            total += layer[j] * weights[j][i];
        }
        next_layer[i] = sigmoid_function(total);
    }
    return next_layer;
}


void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, double* const target_output) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    double* output_error = create_loss(network_state->output_layer, target_output, output_size);
    double** hidden_errors = (double**)malloc(hidden_amount * sizeof(double*));
    hidden_errors[0] = create_error(network_state->hidden_layers[hidden_amount - 1], hidden_size, network_state->output_weights, output_error, output_size);
    for(register int i = 1; i < hidden_amount; ++i) {
        const int index = hidden_amount - 1 - i;
        hidden_errors[i] = create_error(network_state->hidden_layers[index], hidden_size, network_state->hidden_weights[index], hidden_errors[i - 1], hidden_size);
    }

    // Current weights are updated based on the errors calculated from the weights and layers
    // that were saved into network_state during the forward propagation step.
    double* layer, *error;
    int layer_size, error_size;
    double** weights, **delta_weights;

    layer = network_state->hidden_layers[hidden_amount - 1];
    layer_size = hidden_size;
    weights = neural_network->output_weights;
    delta_weights = neural_network->delta_output_weights;
    error = output_error;
    error_size = output_size;
    update_weights(neural_network, layer, layer_size, weights, delta_weights, output_error, output_size);

    for(register int i = 1; i < hidden_amount; ++i) {
        int index = hidden_amount - 1 - i;
        layer = network_state->hidden_layers[index];
        layer_size = hidden_size;
        weights = neural_network->hidden_weights[index];
        delta_weights = neural_network->delta_hidden_weights[index];
        error = hidden_errors[i];
        error_size = hidden_size;
        update_weights(neural_network, layer, layer_size, weights, delta_weights, output_error, output_size);
    }

    layer = network_state->input_layer;
    layer_size = input_size;
    weights = neural_network->input_weights;
    delta_weights = neural_network->delta_input_weights;
    error = hidden_errors[0];
    error_size = hidden_size;
    update_weights(neural_network, layer, layer_size, weights, delta_weights, output_error, output_size);

    free_2D_double_array(hidden_errors, hidden_amount);
    free(output_error);
}


double* create_loss(double* const output, double* const target, const int output_size) {
    double* loss = (double*)malloc(output_size * sizeof(double));
    for(register int i = 0; i < output_size; ++i) {
        loss[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);
    }
    return loss;
}


double* create_error(double* const layer, const int layer_size, double** const weights, double* const previous_error, const int previous_error_size) {
    double* error = (double*)malloc(layer_size * sizeof(double));
    for(register int i = 0; i < layer_size; ++i) {
        double sum = 0;
        for(register int j = 0; j < previous_error_size; ++j) {
            sum += weights[i][j] * previous_error[j];
        }
        error[i] = layer[i] * (1 - layer[i]) * sum;
    }
    return error;
}


void update_weights(struct NeuralNetwork* neural_network, double* const layer, const int layer_size, double** weights, double** delta_weights, double* const error, const int error_size) {
    double learning_rate = neural_network->learning_rate;
    double momentum_value = neural_network->momentum_value;
    int momentum_enabled = neural_network->momentum_enabled;

    for(register int i = 0; i < layer_size; ++i) {
        for(register int j = 0; j < error_size; ++j) {
            double new_value = weights[i][j] + (learning_rate * layer[i] * error[j]);
            new_value += momentum_enabled ? momentum_value * delta_weights[i][j] : 0;
            delta_weights[i][j] = weights[i][j] - new_value;
            weights[i][j] = new_value;
        }
    }
}


struct History* create_history() {
    struct History* new_history = (struct History*)malloc(sizeof(struct History));
    new_history->size = 0;
    new_history->event_head = NULL;
    return new_history;
}


void free_history(struct History* history) {
    if(history) {
        if(history->event_head) {
            free_event(history->event_head);
        }
        free(history);
    }
    history = NULL;
}


void add_event(struct History* history, struct NetworkState* network_state, int chosen_action, double reward) {
    struct Event* new_event = (struct Event*)malloc(sizeof(struct Event));
    new_event->network_state = network_state;
    new_event->chosen_action = chosen_action;
    new_event->reward = reward;
    new_event->next_event = history->event_head;
    history->event_head = new_event;
    ++history->size;
}


void free_event(struct Event* event) {
    if(event) {
        if(event->network_state) {
            free_network_state(event->network_state);
        }
        if(event->next_event) {
            free_event(event->next_event);
        }
        free(event);
    }
    event = NULL;
}


void perform_batch_update_pop_last(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma) {
    struct Event* event = history->event_head;
    struct Event* previous_event = NULL;
    double max_next_state_Qvalue = 0;

    for(register int i = 0; i < history->size; ++i) {
        struct NetworkState* network_state = event->network_state;
        int chosen_action = event->chosen_action;
        int reward = event->reward;

        double* target_output = create_double_array_copy(network_state->output, network_state->output_size);

        /*
        if(i == 0) {
            //target_output[chosen_action] += alpha * reward;
            //target_output[chosen_action] = alpha * reward;
            //target_output[chosen_action] = alpha * (reward + target_output[chosen_action]);
        } else {
            //target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - target_output[chosen_action]);
            //target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - get_max_value(network_state->output, network_state->output_size));
            //target_output[chosen_action] = alpha * (reward + (gamma * next_state_max_Qvalue) - get_max_value(network_state->output, network_state->output_size));
            //target_output[chosen_action] = alpha * (reward + (gamma * next_state_max_Qvalue) - target_output[chosen_action]);
        }
        */

        target_output[chosen_action] = ((1 - alpha) * target_output[chosen_action]) + (alpha * (reward + (gamma * max_next_state_Qvalue)));

        max_next_state_Qvalue = get_max_value(target_output, network_state->output_size);
        double* target = create_sigmoid_array(target_output, network_state->output_size);
        execute_back_propagation(neural_network, network_state, target);
        free(target_output);
        free(target);

        if(i == history->size - 1) {
            if(previous_event) {
                previous_event->next_event = NULL;
            } else {
                history->event_head = NULL;
            }
            free_event(event);
            --history->size;
            break;
        } else {
            previous_event = event;
            event = event->next_event;
        }
    }
}


void perform_batch_update_pop_all(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma) {
    if(history->event_head) {
        preform_batch_update_rec(neural_network, history->event_head, 1, 0, alpha, gamma);
    }
    history->event_head = NULL;
    history->size = 0;
}


void preform_batch_update_rec(struct NeuralNetwork* neural_network, struct Event* event, short is_first_event, double max_next_state_Qvalue, const double alpha, const double gamma) {
    if(!event) {
        return;
    }

    struct NetworkState* network_state = event->network_state;
    int chosen_action = event->chosen_action;
    int reward = event->reward;

    double* target_output = create_double_array_copy(network_state->output, network_state->output_size);

    /*
    if(is_first_event == 1) {
        //target_output[chosen_action] += alpha * reward;
        target_output[chosen_action] = alpha * reward;
        is_first_event = 0;
    } else {
        //target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - target_output[chosen_action]);
        //target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - get_max_value(network_state->output, network_state->output_size));
        target_output[chosen_action] = alpha * (reward + (gamma * previous_max_Qvalue) - get_max_value(network_state->output, network_state->output_size));
    }
    */

    target_output[chosen_action] = ((1 - alpha) * target_output[chosen_action]) + (alpha * (reward + (gamma * max_next_state_Qvalue)));

    max_next_state_Qvalue = get_max_value(target_output, network_state->output_size);
    double* target = create_sigmoid_array(target_output, network_state->output_size);
    execute_back_propagation(neural_network, network_state, target);
    free(target_output);
    free(target);

    preform_batch_update_rec(neural_network, event->next_event, is_first_event, max_next_state_Qvalue, alpha, gamma);
    event->next_event = NULL;
    free_event(event);
}

/*
double sigmoid_function(double x) {
    return 1 / (1 + exp(-x));
}


double inv_sigmoid_function(double x) {
    if(x <= 0) {
        return -700;
    } else if(x >= 1) {
        return 40;
    }
    return log(x) - log(1 - x);
}
*/


double sigmoid_function(double x) {
    if(x < -700) {
        return 0;
    } else if(x > 36) {
        return 1;
    }
    return 1 / (1 + exp(-x));
}


double inv_sigmoid_function(double x) {
    if(x < 1.0E-304) {
        return -700;
    } else if(x > 0.999999999999999) {
        return 36;
    }
    return log(x) - log(1 - x);
}


double* create_sigmoid_array(double* const array, const int size) {
    double* sigmoid_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        sigmoid_array[i] = (double)sigmoid_function(array[i]);
    }
    return sigmoid_array;
}


double* create_inv_sigmoid_array(double* const array, const int size) {
    double* inv_sigmoid_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        inv_sigmoid_array[i] = (double)inv_sigmoid_function(array[i]);
    }
    return inv_sigmoid_array;
}


double get_max_value(double* const array, const int size) {
    double max_value = array[0];
    for(register int i = 1; i < size; ++i) {
        max_value = array[i] > max_value ? array[i] : max_value;
    }
    return max_value;
}


int get_max_index(double* const array, const int size) {
    double max_value = array[0];
    int max_index = 0;
    for(register int i = 1; i < size; ++i) {
        if(array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}


double get_random_double(double min, double max) {
    if(min > max) {
        return 0; 
    } else if(min == max) {
        return min;
    }
    return (double)((((double)rand()/(double)(RAND_MAX)) * (max - min)) + min);
}


double* create_double_array(const int size, const double initial_value) {
    double* new_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        //new_array[i] = initial_value;
        new_array[i] = get_random_double(-0.5, 0.5);
    }
    return new_array;
}


double** create_2D_double_array(const int i_size, const int j_size, const double initial_value) {
    double** new_array = (double**)malloc(i_size * sizeof(double*));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_double_array(j_size, initial_value);
    }
    return new_array;
}


double*** create_3D_double_array(const int i_size, const int j_size, const int k_size, const double initial_value) {
    double*** new_array = (double***)malloc(i_size * sizeof(double**));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_2D_double_array(j_size, k_size, initial_value);
    }
    return new_array;
}


double* create_double_array_copy(double* const array, const int size) {
    double* new_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        new_array[i] = (double)array[i];
    }
    return new_array;
}


double** create_2D_double_array_copy(double** const array, const int i_size, const int j_size) {
    double** new_array = (double**)malloc(i_size * sizeof(double*));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_double_array_copy(array[i], j_size);
    }
    return new_array;
}


double*** create_3D_double_array_copy(double*** const array, const int i_size, const int j_size, const int k_size) {
    double*** new_array = (double***)malloc(i_size * sizeof(double**));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_2D_double_array_copy(array[i], j_size, k_size);
    }
    return new_array;
}


void free_2D_double_array(double** array, const int size) {
    if(array) {
        for(register int i = 0; i < size; ++i) {
            free(array[i]);
        }
        free(array);
    }
    array = NULL;
}


void free_3D_double_array(double*** array, const int i_size, const int j_size) {
    if(array) {
        for(register int i = 0; i < i_size; ++i) {
            if(array[i]) {
                free_2D_double_array(array[i], j_size);
            }
        }
        free(array);
    }
    array = NULL;
}


// Getters used for Python interface...
int get_history_size(struct History* history) {
    return history->size;
}


int get_input_size(struct NetworkState* const network_state) {
    return network_state->input_size;
}


int get_hidden_amount(struct NetworkState* const network_state) {
    return network_state->hidden_amount;
}


int get_hidden_size(struct NetworkState* const network_state) {
    return network_state->hidden_size;
}


int get_output_size(struct NetworkState* const network_state) {
    return network_state->output_size;
}


double* get_input_layer(struct NetworkState* const network_state) {
    return network_state->input_layer;
}


double** get_hidden_layers(struct NetworkState* const network_state) {
    return network_state->hidden_layers;
}


double* get_output_layer(struct NetworkState* const network_state) {
    return network_state->output_layer;
}


double* get_output(struct NetworkState* const network_state) {
    return network_state->output;
}


int choose_action(struct NetworkState* const network_state) {
    return get_max_index(network_state->output, network_state->output_size);
}


// For testing...

void display_output(struct NetworkState* const network_state) {
    printf("[");
    for(register int i = 0; i < network_state->output_size; ++i) {
        printf(" %lf ", network_state->output[i]);
    }
    printf("]");
    printf("\t");
    printf("[");
    for(register int i = 0; i < network_state->output_size; ++i) {
        printf(" %lf ", network_state->output_layer[i]);
    }
    printf("]");
    fflush(stdout);
}


int main() {
    printf("Starting...\n");

    struct NeuralNetwork* neural_network = create_network(210 * 160 * 2, 1, 50, 4, 0.1, 0.9, 0);
    struct History* history = create_history();

    for(int i = 0; i < 1000; ++i) {
        double input[210 * 160 * 2];
        for(register int j = 0; j < 210 * 160 * 2; ++j) {
            input[j] = 255;
        }

        struct NetworkState* network_state = execute_forward_propagation(neural_network, input);
        double* output = get_output(network_state);
        int chosen_action = get_max_index(output, get_output_size(network_state));
        double reward = -10;

        if(i > 75) {
            reward = 10;
        }

        printf("Step: %d, [", i + 1);
        display_output(network_state);
        printf("], action: %d, reward: %lf\n", chosen_action, reward);

        add_event(history, network_state, chosen_action, reward);
        
        if(get_history_size(history) >= 30) {
            perform_batch_update_pop_last(neural_network, history, 0.5, 0.9);
        }
    }

    while(history->size > 0) {
        perform_batch_update_pop_last(neural_network, history, 0.5, 0.9);
    }

    free_history(history);
    free_neural_network(neural_network);

    printf("Finished.\n");

    return 0;
}
