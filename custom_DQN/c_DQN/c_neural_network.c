#include "c_DQN.h"


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
        next_layer[i] = sigmoid_function((double)(total));
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
        long double sum = 0;
        for(register int j = 0; j < previous_error_size; ++j) {
            sum += weights[i][j] * previous_error[j];
        }
        error[i] = layer[i] * (1 - layer[i]) * (double)(sum);
    }
    return error;
}


void update_weights(struct NeuralNetwork* neural_network, double* const layer, const int layer_size, double** weights, double** delta_weights, double* const error, const int error_size) {
    double learning_rate = neural_network->learning_rate;
    double momentum_value = neural_network->momentum_value;
    int momentum_enabled = neural_network->momentum_enabled;

    for(register int i = 0; i < layer_size; ++i) {
        for(register int j = 0; j < error_size; ++j) {
            long double new_value = weights[i][j] + (learning_rate * layer[i] * error[j]);
            new_value += momentum_enabled ? momentum_value * delta_weights[i][j] : 0;
            delta_weights[i][j] = weights[i][j] - (double)(new_value);
            weights[i][j] = (double)(new_value);
        }
    }
}


