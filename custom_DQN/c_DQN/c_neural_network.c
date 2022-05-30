#include "c_DQN.h"


struct NeuralNetwork* create_neural_network(int input_size, int hidden_amount, int hidden_size, int output_size, double learning_rate, double momentum_value, int momentum_enabled, int randomize_weights) {
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
    randomize_weights = randomize_weights == 0 ? 0 : 1;
    neural_network->input_weights = create_2D_double_array(input_size + 1, hidden_size, randomize_weights);
    neural_network->hidden_weights = create_3D_double_array(hidden_amount - 1, hidden_size + 1, hidden_size, randomize_weights);
    neural_network->output_weights = create_2D_double_array(hidden_size + 1, output_size, randomize_weights);

    // Allocating delta weights.
    neural_network->delta_input_weights = create_2D_double_array(input_size + 1, hidden_size, 0);
    neural_network->delta_hidden_weights = create_3D_double_array(hidden_amount - 1, hidden_size + 1, hidden_size, 0);
    neural_network->delta_output_weights = create_2D_double_array(hidden_size + 1, output_size, 0);

    return neural_network;
}


void free_neural_network(struct NeuralNetwork* neural_network) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    if(neural_network) {
        if(neural_network->input_weights) {
            free_2D_double_array(neural_network->input_weights, input_size + 1);
        }
        if(neural_network->hidden_weights) {
            free_3D_double_array(neural_network->hidden_weights, hidden_amount - 1, hidden_size + 1);
        }
        if(neural_network->output_weights) {
            free_2D_double_array(neural_network->output_weights, hidden_size + 1);
        }
        if(neural_network->delta_input_weights) {
            free_2D_double_array(neural_network->delta_input_weights, input_size + 1);
        }
        if(neural_network->delta_hidden_weights) {
            free_3D_double_array(neural_network->delta_hidden_weights, hidden_amount - 1, hidden_size + 1);
        }
        if(neural_network->delta_output_weights) {
            free_2D_double_array(neural_network->delta_output_weights, hidden_size + 1);
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
        free(network_state);
    }
    network_state = NULL;
}


struct NetworkState* execute_forward_propagation(struct NeuralNetwork* const neural_network, struct ConvLayer* conv_layer) {
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
    network_state->input_layer = create_input_layer(conv_layer);
    network_state->hidden_layers = (double**)malloc(hidden_amount * sizeof(double*));
    network_state->hidden_layers[0] = create_next_layer(network_state->input_layer, input_size, neural_network->input_weights, hidden_size);
    for(register int i = 1; i < hidden_amount; ++i) {
        network_state->hidden_layers[i] = create_next_layer(network_state->hidden_layers[i - 1], hidden_size, neural_network->hidden_weights[i - 1], hidden_size);
    }
    network_state->output_layer = create_next_layer(network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, output_size);
    
    return network_state;
}


struct CreateNextLayerThreadParams {
    int starting_index;
    int job_size;
    double* layer;
    int layer_size;
    double** weights;
    double* next_layer;
};


void* create_next_layer_thread(void* params_ptr) {
    struct CreateNextLayerThreadParams* params = (struct CreateNextLayerThreadParams*)params_ptr;
    int starting_index = params->starting_index;
    int job_size = params->job_size;
    double* layer = params->layer;
    int layer_size = params->layer_size;
    double** weights = params->weights;
    double* next_layer = params->next_layer;

    for(register int i = starting_index; i < starting_index + job_size; ++i) {
        long double sum = 0;
        for(register int j = 0; j < layer_size; ++j) {
            sum += layer[j] * weights[j][i];
        }
        next_layer[i] = sigmoid_function(sum);
    }
}


double* create_next_layer(double* const layer, const int layer_size, double** const weights, const int next_layer_size) {
    int biased_layer_size = layer_size + 1;
    double biased_layer[biased_layer_size];
    get_biased_array(biased_layer, layer, layer_size);

    double* next_layer = (double*)malloc(next_layer_size * sizeof(double));

    int thread_amount = (next_layer_size < MAX_THREADS) ? 1 : MAX_THREADS;
    int job_size = (int)(next_layer_size / thread_amount);
    int last_job_size = next_layer_size - (job_size * (thread_amount - 1));
    struct CreateNextLayerThreadParams params_list[thread_amount];
    pthread_t threads[thread_amount];
    for(register int i = 0; i < thread_amount; ++i) {
        params_list[i].starting_index = job_size * i;
        params_list[i].job_size = i == thread_amount - 1 ? last_job_size : job_size;
        params_list[i].layer = biased_layer;
        params_list[i].layer_size = biased_layer_size;
        params_list[i].weights = weights;
        params_list[i].next_layer = next_layer;
        pthread_create(&threads[i], NULL, create_next_layer_thread, &params_list[i]);
    }
    for(register int i = 0; i < thread_amount; ++i) {
        pthread_join(threads[i], NULL);
    }

    return next_layer;
}


void execute_back_propagation(struct NeuralNetwork* const neural_network, struct NetworkState* const network_state, double* const target_output) {
    const int input_size = neural_network->input_size;
    const int hidden_amount = neural_network->hidden_amount;
    const int hidden_size = neural_network->hidden_size;
    const int output_size = neural_network->output_size;

    // The output and hidden errors are stack allocated below, 
    // so if the size or amount of layers is too large, this will seg-fault. 
    // Use malloc instead if this is the case.

    long double output_error[output_size];
    get_loss(output_error, network_state->output_layer, target_output, output_size);

    long double hidden_errors[hidden_amount][hidden_size];
    get_error(hidden_errors[hidden_amount - 1], network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, output_error, output_size);

    for(register int i = 0; i < hidden_amount - 1; ++i) {
        int index = hidden_amount - 2 - i;
        get_error(hidden_errors[index], network_state->hidden_layers[index], hidden_size, neural_network->hidden_weights[index], hidden_errors[index + 1], hidden_size);
    }

    // Current weights are updated based on the errors calculated from the weights and layers
    // that were saved into network_state during the forward propagation step.
    update_weights(neural_network, network_state->hidden_layers[hidden_amount - 1], hidden_size, neural_network->output_weights, neural_network->delta_output_weights, output_error, output_size);
    for(register int i = 0; i < hidden_amount - 1; ++i) {
        int index = hidden_amount - 2 - i;
        update_weights(neural_network, network_state->hidden_layers[index], hidden_size, neural_network->hidden_weights[index], neural_network->delta_hidden_weights[index], hidden_errors[index + 1], hidden_size);
    }
    update_weights(neural_network, network_state->input_layer, input_size, neural_network->input_weights, neural_network->delta_input_weights, hidden_errors[0], hidden_size);
}


void get_loss(long double* loss, double* const output, double* const target, const int output_size) {
    for(register int i = 0; i < output_size; ++i) {
        loss[i] = (long double)output[i] * (1 - (long double)output[i]) * ((long double)output[i] - (long double)target[i]);
        //loss[i] = (1 - ((long double)output[i] * (long double)output[i])) * ((long double)output[i] - (long double)target[i]);
        //loss[i] = (1 - ((long double)output[i] * (long double)output[i])) * ((long double)target[i] - (long double)output[i]);
    }
}


struct GetErrorThreadParams {
    int starting_index;
    int job_size;
    double* layer;
    double** weights; 
    long double* previous_error; 
    int previous_error_size;
    long double* error;
};


void* get_error_thread(void* params_ptr) {
    struct GetErrorThreadParams* params = (struct GetErrorThreadParams*)params_ptr;
    int starting_index = params->starting_index;
    int job_size = params->job_size;
    double* layer = params->layer;
    double** weights = params->weights; 
    long double* previous_error = params->previous_error; 
    int previous_error_size = params->previous_error_size;
    long double* error = params->error;

    for(register int i = starting_index; i < starting_index + job_size; ++i) {
        long double sum = 0;
        for(register int j = 0; j < previous_error_size; ++j) {
            sum += (long double)weights[i][j] * (long double)previous_error[j];
        }
        error[i] = (long double)layer[i] * (1 - (long double)layer[i]) * (long double)sum;
        //error[i] = (1 - ((long double)layer[i] * (long double)layer[i])) * (long double)(sum);
    }
}


void get_error(long double* error, double* const layer, const int layer_size, double** const weights, long double* const previous_error, const int previous_error_size) {
    int thread_amount = layer_size < MAX_THREADS ? 1 : MAX_THREADS;
    int job_size = (int)(layer_size / thread_amount);
    int last_job_size = layer_size - (job_size * (thread_amount - 1));
    struct GetErrorThreadParams params_list[thread_amount];
    pthread_t threads[thread_amount];
    for(register int i = 0; i < thread_amount; ++i) {
        params_list[i].starting_index = job_size * i;
        params_list[i].job_size = i == thread_amount - 1 ? last_job_size : job_size;
        params_list[i].layer = layer;
        params_list[i].weights = weights; 
        params_list[i].previous_error = previous_error; 
        params_list[i].previous_error_size = previous_error_size;
        params_list[i].error = error;
        pthread_create(&threads[i], NULL, get_error_thread, &params_list[i]);
    }
    for(register int i = 0; i < thread_amount; ++i) {
        pthread_join(threads[i], NULL);
    }
}


struct UpdateWeightsThreadParams {
    int starting_index;
    int job_size;
    double* layer;
    double** weights;
    double** delta_weights;
    long double* error;
    int error_size;
    long double learning_rate;
    long double momentum_value;
    int momentum_enabled;
};


void* update_weights_thread(void* params_ptr) {
    struct UpdateWeightsThreadParams* params = (struct UpdateWeightsThreadParams*)params_ptr;
    int starting_index = params->starting_index;
    int job_size = params->job_size;
    double* layer = params->layer;
    double** weights = params->weights;
    double** delta_weights = params->delta_weights;
    long double* error = params->error;
    int error_size = params->error_size;
    long double learning_rate = params->learning_rate;
    long double momentum_value = params->momentum_value;
    int momentum_enabled = params->momentum_enabled;

    for(register int i = starting_index; i < starting_index + job_size; ++i) {
        for(register int j = 0; j < error_size; ++j) {
            long double new_value = (long double)weights[i][j] - ((long double)learning_rate * (long double)layer[i] * (long double)error[j]);
            new_value += momentum_enabled ? ((long double)delta_weights[i][j] * (long double)momentum_value) : 0;
            delta_weights[i][j] = new_value - (long double)weights[i][j];
            weights[i][j] = new_value;
        }
    }
}


void update_weights(struct NeuralNetwork* neural_network, double* const layer, const int layer_size, double** weights, double** delta_weights, long double* const error, const int error_size) {
    int biased_layer_size = layer_size + 1;
    double biased_layer[biased_layer_size];
    get_biased_array(biased_layer, layer, layer_size);

    int thread_amount = biased_layer_size < MAX_THREADS ? 1 : MAX_THREADS;
    int job_size = (int)(biased_layer_size / thread_amount);
    int last_job_size = biased_layer_size - (job_size * (thread_amount - 1));
    struct UpdateWeightsThreadParams params_list[thread_amount];
    pthread_t threads[thread_amount];
    for(register int i = 0; i < thread_amount; ++i) {
        params_list[i].starting_index = job_size * i;
        params_list[i].job_size = i == thread_amount - 1 ? last_job_size : job_size;
        params_list[i].layer = biased_layer;
        params_list[i].weights = weights;
        params_list[i].delta_weights = delta_weights;
        params_list[i].error = error;
        params_list[i].error_size = error_size;
        params_list[i].learning_rate = neural_network->learning_rate;
        params_list[i].momentum_value = neural_network->momentum_value;
        params_list[i].momentum_enabled = neural_network->momentum_enabled;
        pthread_create(&threads[i], NULL, update_weights_thread, &params_list[i]);
    }
    for(register int i = 0; i < thread_amount; ++i) {
        pthread_join(threads[i], NULL);
    }
}