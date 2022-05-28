#include "c_DQN.h"


long double sigmoid_function(long double x) {
   /*
    if(x >= 18) {
        return 1;
    }
    if(x <= -18) {
        return -1;
    }
    return (long double)(((long double)exp(2 * x) - 1) / ((long double)exp(2 * x) + 1));
    */
    //if(x <= -700) {
    //    return 0;
    //} else if(x >= 36) {
    //    return 1;
    //}

    //return (long double)(1 / (1 + (long double)exp(-x)));
    //return x / (1 + absolute_value(x));
    //return 0.5 * (x / (1 + absolute_value(x)) + 1);

    long double result = (long double)(1 / (1 + (long double)exp(-x)));
    if(!isfinite(result)) {
        result = 1 / (1 + exp(-((float)x)));
        if(!isfinite(result)) {
            result = x > 0 ? 1 : 0;
        }
    }
    return result;
}


long double inv_sigmoid_function(long double x) {
    /*
    if(x > 0.9999999999999995) {
        return 18;
    }
    if(x < -0.9999999999999995) {
        return -18;
    }
    return (long double)(0.5 * ((long double)log(1 + x) - (long double)log(1 - x)));
    */
    //if(x < 1E-304) {
    //    return -700;
    //} else if(x > 0.999999999999999) {
    //    return 36;
    //}
    //return (long double)((long double)log(x) - (long double)log(1 - x));
    long double result = (long double)log(x / (1 - x));
    if(!isfinite(result)) {
        result = log(x / (1 - ((float)x)));
        if(!isfinite(result)) {
            result = x > 0.5 ? 45 : -710;
        }
    }
    return result;
}


void apply_sigmoid_to_array(double* dest_array, double* src_array, const int size) {
    for(register int i = 0; i < size; ++i) {
        dest_array[i] = (double)sigmoid_function(src_array[i]);
    }
}


double* create_sigmoid_array(double* const array, const int size) {
    double* sigmoid_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        sigmoid_array[i] = (double)sigmoid_function(array[i]);
    }
    return sigmoid_array;
}


void apply_inv_sigmoid_to_array(double* dest_array, double* src_array, const int size) {
    for(register int i = 0; i < size; ++i) {
        dest_array[i] = (double)inv_sigmoid_function(src_array[i]);
    }
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
    return min + ((long double)rand() / RAND_MAX) * (max - min);
}


double* create_double_array(const int size, const int randomize) {
    double* new_array = (double*)malloc(size * sizeof(double));
    for(register int i = 0; i < size; ++i) {
        new_array[i] = randomize ? get_random_double(-0.05, 0.05) : 0;
    }
    return new_array;
}


double** create_2D_double_array(const int i_size, const int j_size, const int randomize) {
    double** new_array = (double**)malloc(i_size * sizeof(double*));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_double_array(j_size, randomize);
    }
    return new_array;
}


double*** create_3D_double_array(const int i_size, const int j_size, const int k_size, const int randomize) {
    double*** new_array = (double***)malloc(i_size * sizeof(double**));
    for(register int i = 0; i < i_size; ++i) {
        new_array[i] = create_2D_double_array(j_size, k_size, randomize);
    }
    return new_array;
}


void copy_double_array(double* dest_array, double* const src_array, const int size) {
    for(register int i = 0; i < size; ++i) {
        dest_array[i] = (double)src_array[i];
    }
}


void copy_2D_double_array(double** dest_array, double** const src_array, const int i_size, const int j_size) {
    for(register int i = 0; i < i_size; ++i) {
        copy_double_array(dest_array[i], src_array[i], j_size);
    }
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


void get_biased_array(double* dest_array, double* const src_array, const int size) {
    dest_array[size] = (double)BIAS_VALUE;
    for(register int i = 0; i < size; ++i) {
        dest_array[i] = (double)src_array[i];
    }
}


double* create_biased_array_copy(double* const array, const int size) {
    double* new_array = (double*)malloc((size + 1) * sizeof(double));
    new_array[size] = BIAS_VALUE;
    for(register int i = 0; i < size; ++i) {
        new_array[i] = (double)array[i];
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


void free_2D_long_double_array(long double** array, const int size) {
    if(array) {
        for(register int i = 0; i < size; ++i) {
            free(array[i]);
        }
        free(array);
    }
    array = NULL;
}


void free_3D_long_double_array(long double*** array, const int i_size, const int j_size) {
    if(array) {
        for(register int i = 0; i < i_size; ++i) {
            if(array[i]) {
                free_2D_long_double_array(array[i], j_size);
            }
        }
        free(array);
    }
    array = NULL;
}


void display_output(struct NetworkState* const network_state) {
    printf("[");
    for(register int i = 0; i < network_state->output_size; ++i) {
        printf(" %lf ", network_state->output_layer[i]);
    }
    printf("]");
    printf("\t");
    double output_values [network_state->output_size];
    apply_inv_sigmoid_to_array(output_values, network_state->output_layer, network_state->output_size);
    printf("[");
    for(register int i = 0; i < network_state->output_size; ++i) {
        printf(" %lf ", output_values[i]);
    }
    printf("]");
    fflush(stdout);
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


int choose_action(struct NetworkState* const network_state) {
    return get_max_index(network_state->output_layer, network_state->output_size);
}