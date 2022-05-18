#include "c_DQN.h"


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


void perform_batch_update_pop_amount(struct NeuralNetwork* neural_network, struct History* history, int pop_amount, const double alpha, const double gamma) {
    struct Event* event = history->event_head;
    struct Event* last_event = NULL;
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

        if(i == history->size - pop_amount - 1) {
            last_event = event;
        }

        event = event->next_event;
    }
    if(last_event) {
        free_event(last_event->next_event);
        last_event->next_event = NULL;
    } else {
        free_event(history->event_head);
        history->event_head = NULL;
    }
    history->size -= pop_amount;
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
