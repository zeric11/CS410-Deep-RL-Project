// Compile to use with Python:  gcc -fPIC -shared -o c_neural_network.so c_neural_network.c
// Compile for testing:         gcc -g c_neural_network.c -o c_neural_network -lm 
// Check for mem-leaks:         valgrind --tool=memcheck --leak-check=yes -s ./c_neural_network


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


const float LEARNING_RATE = 0.5;
const float MOMENTUM_ENABLED = 0;
const float MOMENTUM_VALUE = 0.9;


struct NetworkState {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    float* input_layer;
    float** hidden_layers;
    float* output_layer;

    float** input_weights;
    float*** hidden_weights;
    float** output_weights;
};


struct NeuralNetwork {
    int input_size;
    int hidden_amount;
    int hidden_size;
    int output_size;

    float** input_weights;
    float*** hidden_weights;
    float** output_weights;

    float** delta_input_weights;
    float*** delta_hidden_weights;
    float** delta_output_weights;
};


struct Event {
    struct NetworkState* network_state;
    int chosen_action;
    float reward;
    struct Event* next_event;
};


struct History {
    int size;
    struct Event* event_head;
};


struct NeuralNetwork* create_network(const int input_size, const int hidden_amount, const int hidden_size, const int output_size);
void free_neural_network(struct NeuralNetwork* neural_network);

struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, float* const input);
float* create_next_layer(float* const layer, const int layer_size, float** const weights, const int next_layer_size);
void free_network_state(struct NetworkState* network_state);

void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, float* const target_output);
float* create_loss(float* const output, float* const target_output, const int output_size);
float* create_error(float* const layer, const int layer_size, float** const weights, float* const previous_error, const int previous_error_size);
void update_weights(float* const layer, const int layer_size, float** weights, float** delta_weights, float* const error, const int error_size);

struct History* create_history();
void free_history(struct History* history);
void add_event(struct History* history, struct NetworkState* network_state, int chosen_action, float reward);
void free_event(struct Event* event);
void perform_batch_update(struct NeuralNetwork* neural_network, struct History* history, const float alpha, const float gamma);
void preform_batch_update_rec(struct NeuralNetwork* neural_network, struct Event* event, short is_first_event, float previous_max_Qvalue, const float alpha, const float gamma);

double sigmoid_function(double x);
double inv_sigmoid_function(double x);
float* create_sigmoid_array(float* const array, const int size);
float* create_inv_sigmoid_array(float* const array, const int size);
float get_max_value(float* const array, const int size);

float* create_float_array(const int size, const float initial_value);
float** create_2D_float_array(const int i_size, const int j_size, const float initial_value);
float*** create_3D_float_array(const int i_size, const int j_size, const int k_size, const float initial_value);

float* create_float_array_copy(float* const array, const int size);
float** create_2D_float_array_copy(float** const array, const int i_size, const int j_size);
float*** create_3D_float_array_copy(float*** const array, const int i_size, const int j_size, const int k_size);

void free_2D_float_array(float** array, const int size);
void free_3D_float_array(float*** array, const int i_size, const int j_size);

int get_history_size(struct History* history);
int get_input_size(struct NetworkState* const network_state);
int get_hidden_amount(struct NetworkState* const network_state);
int get_hidden_size(struct NetworkState* const network_state);
int get_output_size(struct NetworkState* const network_state);
float* get_input_layer(struct NetworkState* const network_state);
float** get_hidden_layers(struct NetworkState* const network_state);
float* get_output_layer(struct NetworkState* const network_state);


struct NeuralNetwork* create_network(const int input_size, const int hidden_amount, const int hidden_size, const int output_size) {
    struct NeuralNetwork* neural_network = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));

    neural_network->input_size = input_size;
    neural_network->hidden_amount = hidden_amount;
    neural_network->hidden_size = hidden_size;
    neural_network->output_size = output_size;

    // Allocating weights.
    neural_network->input_weights = create_2D_float_array(input_size, hidden_size, 0);
    neural_network->hidden_weights = create_3D_float_array(hidden_amount - 1, hidden_size, hidden_size, 0);
    neural_network->output_weights = create_2D_float_array(hidden_size, output_size, 0);

    // Allocating delta weights.
    neural_network->delta_input_weights = create_2D_float_array(input_size, hidden_size, 0);
    neural_network->delta_hidden_weights = create_3D_float_array(hidden_amount - 1, hidden_size, hidden_size, 0);
    neural_network->delta_output_weights = create_2D_float_array(hidden_size, output_size, 0);

    return neural_network;
}


void free_neural_network(struct NeuralNetwork* neural_network) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    if(neural_network) {
        if(neural_network->input_weights) {
            free_2D_float_array(neural_network->input_weights, input_size);
        }
        if(neural_network->hidden_weights) {
            free_3D_float_array(neural_network->hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(neural_network->output_weights) {
            free_2D_float_array(neural_network->output_weights, hidden_size);
        }
        if(neural_network->delta_input_weights) {
            free_2D_float_array(neural_network->delta_input_weights, input_size);
        }
        if(neural_network->delta_hidden_weights) {
            free_3D_float_array(neural_network->delta_hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(neural_network->delta_output_weights) {
            free_2D_float_array(neural_network->delta_output_weights, hidden_size);
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
            free_2D_float_array(network_state->hidden_layers, hidden_amount);
        }
        if(network_state->output_layer) {
            free(network_state->output_layer);
        }
        if(network_state->input_weights) {
            free_2D_float_array(network_state->input_weights, input_size);
        }
        if(network_state->hidden_weights) {
            free_3D_float_array(network_state->hidden_weights, hidden_amount - 1, hidden_size);
        }
        if(network_state->output_weights) {
            free_2D_float_array(network_state->output_weights, hidden_size);
        }
        free(network_state);
    }
    network_state = NULL;
}


struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, float* const input) {
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
    network_state->input_layer = create_float_array_copy(input, input_size);
    network_state->hidden_layers = (float**)malloc(hidden_amount * sizeof(float*));
    network_state->hidden_layers[0] = create_next_layer(input, input_size, neural_network->input_weights, hidden_size);
    for(register int i = 0; i < hidden_amount - 1; ++i) {
        network_state->hidden_layers[i + 1] = create_next_layer(network_state->hidden_layers[i], hidden_size, neural_network->hidden_weights[i], hidden_size);
    }
    network_state->output_layer = create_next_layer(network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, output_size);

    network_state->input_weights = create_2D_float_array_copy(neural_network->input_weights, input_size, hidden_size);
    network_state->hidden_weights = create_3D_float_array_copy(neural_network->hidden_weights, hidden_amount - 1, hidden_size, hidden_size);
    network_state->output_weights = create_2D_float_array_copy(neural_network->output_weights, hidden_size, output_size);
    
    return network_state;
}


float* create_next_layer(float* const layer, const int layer_size, float** const weights, const int next_layer_size) {
    float* next_layer = (float*)malloc(next_layer_size * sizeof(float));
    for(register int i = 0; i < next_layer_size; ++i) {
        float total = 0;
        for(register int j = 0; j < layer_size; ++j) {
            total += layer[j] * weights[j][i];
        }
        next_layer[i] = sigmoid_function(total);
    }
    return next_layer;
}


void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, float* const target_output) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    float* output_error = create_loss(network_state->output_layer, target_output, output_size);
    float** hidden_errors = (float**)malloc(hidden_amount * sizeof(float*));
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

    free_2D_float_array(hidden_errors, hidden_amount);
    free(output_error);
}


float* create_loss(float* const output, float* const target, const int output_size) {
    float* loss = (float*)malloc(output_size * sizeof(float));
    for(register int i = 0; i < output_size; ++i) {
        loss[i] = output[i] * (1 - output[i]) * (target[i] - output[i]);
    }
    return loss;
}


float* create_error(float* const layer, const int layer_size, float** const weights, float* const previous_error, const int previous_error_size) {
    float* error = (float*)malloc(layer_size * sizeof(float));
    for(register int i = 0; i < layer_size; ++i) {
        float sum = 0;
        for(register int j = 0; j < previous_error_size; ++j) {
            sum += weights[i][j] * previous_error[j];
        }
        error[i] = layer[i] * (1 - layer[i]) * sum;
    }
    return error;
}


void update_weights(float* const layer, const int layer_size, float** weights, float** delta_weights, float* const error, const int error_size) {
    for(register int i = 0; i < layer_size; ++i) {
        for(register int j = 0; j < error_size; ++j) {
            float new_value = weights[i][j] + (LEARNING_RATE * layer[i] * error[j]);
            new_value += MOMENTUM_ENABLED ? MOMENTUM_VALUE * delta_weights[i][j] : 0;
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


void add_event(struct History* history, struct NetworkState* network_state, int chosen_action, float reward) {
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


void perform_batch_update(struct NeuralNetwork* neural_network, struct History* history, const float alpha, const float gamma) {
    if(history->event_head) {
        preform_batch_update_rec(neural_network, history->event_head, 1, 0, alpha, gamma);
    }
    history->event_head = NULL;
    history->size = 0;
}


void preform_batch_update_rec(struct NeuralNetwork* neural_network, struct Event* event, short is_first_event, float previous_max_Qvalue, const float alpha, const float gamma) {
    if(!event) {
        return;
    }

    struct NetworkState* network_state = event->network_state;
    int chosen_action = event->chosen_action;
    int reward = event->reward;

    float* output = create_inv_sigmoid_array(network_state->output_layer, network_state->output_size);
    float* target_output = create_float_array_copy(output, network_state->output_size);
    if(is_first_event == 1) {
        target_output[chosen_action] += alpha * reward;
        is_first_event = 0;
    } else {
        target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - get_max_value(output, network_state->output_size));
        //target_output[chosen_action] += alpha * (reward + (gamma * previous_max_Qvalue) - target_output[chosen_action]);
    }

    previous_max_Qvalue = get_max_value(target_output, network_state->output_size);
    float* target = create_sigmoid_array(target_output, network_state->output_size);
    execute_back_propagation(neural_network, network_state, target);
    free(output);
    free(target_output);
    free(target);

    preform_batch_update_rec(neural_network, event->next_event, is_first_event, previous_max_Qvalue, alpha, gamma);
    event->next_event = NULL;
    free_event(event);
}


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


float* create_sigmoid_array(float* const array, const int size) {
    float* sigmoid_array = (float*)malloc(size * sizeof(float));
    for(register int i = 0; i < size; ++i) {
        sigmoid_array[i] = (float)sigmoid_function(array[i]);
    }
    return sigmoid_array;
}


float* create_inv_sigmoid_array(float* const array, const int size) {
    float* inv_sigmoid_array = (float*)malloc(size * sizeof(float));
    for(register int i = 0; i < size; ++i) {
        inv_sigmoid_array[i] = (float)inv_sigmoid_function(array[i]);
    }
    return inv_sigmoid_array;
}


float get_max_value(float* const array, const int size) {
    float max_value = array[0];
    for(register int i = 1; i < size; ++i) {
        max_value = array[i] > max_value ? array[i] : max_value;
    }
    return max_value;
}


float* create_float_array(const int size, const float initial_value) {
    float* new_array = (float*)malloc(size * sizeof(float));
    for(register int i = 0; i < size; ++i) {
        new_array[i] = initial_value;
    }
    return new_array;
}


float** create_2D_float_array(const int i_size, const int j_size, const float initial_value) {
    float** new_array = (float**)malloc(i_size * sizeof(float*));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_float_array(j_size, initial_value);
    }
    return new_array;
}


float*** create_3D_float_array(const int i_size, const int j_size, const int k_size, const float initial_value) {
    float*** new_array = (float***)malloc(i_size * sizeof(float**));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_2D_float_array(j_size, k_size, initial_value);
    }
    return new_array;
}


float* create_float_array_copy(float* const array, const int size) {
    float* new_array = (float*)malloc(size * sizeof(float));
    for(register int i = 0; i < size; ++i) {
        new_array[i] = array[i];
    }
    return new_array;
}


float** create_2D_float_array_copy(float** const array, const int i_size, const int j_size) {
    float** new_array = (float**)malloc(i_size * sizeof(float*));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_float_array_copy(array[i], j_size);
    }
    return new_array;
}


float*** create_3D_float_array_copy(float*** const array, const int i_size, const int j_size, const int k_size) {
    float*** new_array = (float***)malloc(i_size * sizeof(float**));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_2D_float_array_copy(array[i], j_size, k_size);
    }
    return new_array;
}


void free_2D_float_array(float** array, const int size) {
    if(array) {
        for(register int i = 0; i < size; ++i) {
            free(array[i]);
        }
        free(array);
    }
    array = NULL;
}


void free_3D_float_array(float*** array, const int i_size, const int j_size) {
    if(array) {
        for(register int i = 0; i < i_size; ++i) {
            if(array[i]) {
                free_2D_float_array(array[i], j_size);
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


float* get_input_layer(struct NetworkState* const network_state) {
    return network_state->input_layer;
}


float** get_hidden_layers(struct NetworkState* const network_state) {
    return network_state->hidden_layers;
}


float* get_output_layer(struct NetworkState* const network_state) {
    return network_state->output_layer;
}


/*
// For testing...
int main() {
    printf("Starting...\n");

    struct NeuralNetwork* neural_network = create_network(67200, 1, 100, 4);
    struct History* history = create_history();

    for(int i = 0; i < 100; ++i) {
        float* input = (float*)malloc(67200 * sizeof(float));
        for(int j = 0; j < 67200; ++j) {
            input[j] = j;
        }

        struct NetworkState* network_state = execute_forward_propagation(neural_network, input);
        int chosen_action = i % 4;
        float reward = 10 - chosen_action;
        add_event(history, network_state, chosen_action, reward);
        
        free(input);
    }

    perform_batch_update(neural_network, history, 0.5, 0.9);

    free_history(history);
    free_neural_network(neural_network);

    printf("Finished.\n");

    return 0;
}
*/