// Compile to use with Python:  gcc -fPIC -shared -o neural_network_c.so neural_network_c.c
// Check for mem-leaks:         valgrind --tool=memcheck --leak-check=yes -s ./a.out


#include <stdio.h>
#include <stdlib.h>


const double LEARNING_RATE = 0.1;
const double MOMENTUM_ENABLED = 1;
const double MOMENTUM_VALUE = 0.9;


struct NetworkState {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double* input_layer;
    double** hidden_layers;
    double* output_layer;

    double** input_weights;
    double*** hidden_weights;
    double** output_weights;
};


struct NeuralNetwork {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    double** input_weights;
    double*** hidden_weights;
    double** output_weights;

    double** delta_input_weights;
    double*** delta_hidden_weights;
    double** delta_output_weights;
};


struct NeuralNetwork* create_network(const int input_size, const int hidden_amount, const int hidden_size, const int output_size);
void free_neural_network(struct NeuralNetwork* neural_network);

struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, double* const input);
double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size);
void free_network_state(struct NetworkState* network_state);

void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, double* const target_output);
double* create_loss(double* const output, double* const target_output, const int output_size);
double* create_error(double* const layer, const int layer_size, double** const weights, double* const previous_error, const int previous_error_size);
void update_weights(double* const layer, const int layer_size, double** weights, double** delta_weights, double* const error, const int error_size);

double* create_double_array(const int size, const double initial_value);
double** create_2D_double_array(const int i_size, const int j_size, const double initial_value);
double*** create_3D_double_array(const int i_size, const int j_size, const int k_size, const double initial_value);

double* create_double_array_copy(double* const array, const int size);
double** create_2D_double_array_copy(double** const array, const int i_size, const int j_size);
double*** create_3D_double_array_copy(double*** const array, const int i_size, const int j_size, const int k_size);

void free_2D_double_array(double** array, const int size);
void free_3D_double_array(double*** array, const int i_size, const int j_size);

int get_input_size(struct NetworkState* const network_state);
int get_hidden_amount(struct NetworkState* const network_state);
int get_hidden_size(struct NetworkState* const network_state);
int get_output_size(struct NetworkState* const network_state);
double* get_input_layer(struct NetworkState* const network_state);
double** get_hidden_layers(struct NetworkState* const network_state);
double* get_output_layer(struct NetworkState* const network_state);


struct NeuralNetwork* create_network(const int input_size, const int hidden_amount, const int hidden_size, const int output_size) {
    struct NeuralNetwork* neural_network = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));

    neural_network->input_size = input_size;
    neural_network->hidden_amount = hidden_amount;
    neural_network->hidden_size = hidden_size;
    neural_network->output_size = output_size;

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

    network_state->input_weights = create_2D_double_array_copy(neural_network->input_weights, input_size, hidden_size);
    network_state->hidden_weights = create_3D_double_array_copy(neural_network->hidden_weights, hidden_amount - 1, hidden_size, hidden_size);
    network_state->output_weights = create_2D_double_array_copy(neural_network->output_weights, hidden_size, output_size);
    
    return network_state;
}


double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size) {
    double* next_layer = (double*)malloc(next_layer_size * sizeof(double));
    for(register int i = 0; i < next_layer_size; ++i) {
        double total = 0;
        for(register int j = 0; j < layer_size; ++j) {
            total += layer[j] * weights[j][i];
        }
        next_layer[i] = total / layer_size;
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
    update_weights(network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, neural_network->delta_output_weights, output_error, output_size);
    for(register int i = 1; i < hidden_amount; ++i) {
        int index = hidden_amount - 1 - i;
        update_weights(network_state->hidden_layers[index], hidden_size, neural_network->hidden_weights[index], neural_network->delta_hidden_weights[index], hidden_errors[i], hidden_size);
    }
    update_weights(network_state->input_layer, input_size, neural_network->input_weights, neural_network->delta_input_weights, hidden_errors[0], hidden_size);

    free_2D_double_array(hidden_errors, hidden_amount);
    free(output_error);
}


double* create_loss(double* const output, double* const target_output, const int output_size) {
    double* loss = (double*)malloc(output_size * sizeof(double));
    for(register int i = 0; i < output_size; ++i) {
        loss[i] = output[i] - target_output[i];
    }
    return loss;
}


double* create_error(double* const layer, const int layer_size, double** const weights, double* const previous_error, const int previous_error_size) {
    double* error = (double*)malloc(layer_size * sizeof(double));
    for(register int i = 0; i < layer_size; ++i) {
        double total = 0;
        for(register int j = 0; j < previous_error_size; ++j) {
            //if(previous_error[j] != 0) {
            //    total += previous_error[j] + (layer[i] * weights[i][j]);
            //}
            total += previous_error[j] ? previous_error[j] != 0 + (layer[i] * weights[i][j]) : 0;
        }
        error[i] = total;
    }
    return error;
}


void update_weights(double* const layer, const int layer_size, double** weights, double** delta_weights, double* const error, const int error_size) {
    for(register int i = 0; i < layer_size; ++i) {
        for(register int j = 0; j < error_size; ++j) {
            double new_value = weights[i][j] + (LEARNING_RATE * layer[i] * error[j]);
            new_value += MOMENTUM_ENABLED ? MOMENTUM_VALUE * delta_weights[i][j] : 0;
            delta_weights[i][j] = weights[i][j] - new_value;
            weights[i][j] = new_value;
        }
    }
}


double* create_double_array(const int size, const double initial_value) {
    double* new_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        new_array[i] = initial_value;
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
        new_array[i] = array[i];
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



// For testing...
int main() {
    printf("Starting...\n");

    struct NeuralNetwork* neural_network = create_network(10, 100, 1000, 9);

    double* input = (double*)malloc(10 * sizeof(double));
    for(int i = 0; i < 10; ++i) {
        input[i] = i + 1;
    }

    struct NetworkState* network_state = execute_forward_propagation(neural_network, input);

    double* target = (double*)malloc(9 * sizeof(double));
    for(int i = 0; i < 9; ++i) {
        target[i] = network_state->output_layer[i] + 1;
    }

    execute_back_propagation(neural_network, network_state, target);

    free_neural_network(neural_network);
    free_network_state(network_state);
    free(input);
    free(target);

    printf("Finished.\n");

    return 0;
}
