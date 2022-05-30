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


void perform_batch_update_last(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma) {
    struct Event* event = history->event_head;
    struct Event* last_event = NULL;
    double next_state_max_Qvalue = 0;

    while(event) {
        struct NetworkState* network_state = event->network_state;
        int chosen_action = event->chosen_action;
        int reward = event->reward;

        double target_output[network_state->output_size];
        copy_double_array(target_output, network_state->output_layer, network_state->output_size);

        double chosen_action_Qvalue = inv_sigmoid_function(target_output[chosen_action]);
        double new_Qvalue = ((1 - alpha) * chosen_action_Qvalue) + (alpha * (reward + (gamma * next_state_max_Qvalue)));
        //double new_Qvalue = chosen_action_Qvalue + alpha * (reward + (gamma * next_state_max_Qvalue) - chosen_action_Qvalue);
        if(new_Qvalue != 0) {
            target_output[chosen_action] = sigmoid_function(new_Qvalue);
        }

        if(!event->next_event) {
            execute_back_propagation(neural_network, network_state, target_output);
        } else {
            last_event = event;
        }
        event = event->next_event;
        next_state_max_Qvalue = inv_sigmoid_function(get_max_value(target_output, network_state->output_size));
        //next_state_max_Qvalue = new_Qvalue;
        //next_state_max_Qvalue += reward;
    }
    if(last_event) {
        free_event(last_event->next_event);
        last_event->next_event = NULL;
    } else {
        free_event(history->event_head);
        history->event_head = NULL;
    }
    --history->size;
}


void perform_batch_update_all(struct NeuralNetwork* neural_network, struct History* history, const double alpha, const double gamma) {
    struct Event* event = history->event_head;
    double next_state_max_Qvalue = 0;
    for(register int i = 0; i < history->size; ++i) {
        //events[i] = event;
        struct NetworkState* network_state = event->network_state;
        int chosen_action = event->chosen_action;
        int reward = event->reward;

        double target_output[network_state->output_size];
        copy_double_array(target_output, network_state->output_layer, network_state->output_size);

        double chosen_action_Qvalue = inv_sigmoid_function(target_output[chosen_action]);
        double new_Qvalue = ((1 - alpha) * chosen_action_Qvalue) + (alpha * (reward + (gamma * next_state_max_Qvalue)));
        //double new_Qvalue = chosen_action_Qvalue + alpha * (reward + (gamma * next_state_max_Qvalue) - chosen_action_Qvalue);
        if(new_Qvalue != 0) {
            target_output[chosen_action] = sigmoid_function(new_Qvalue);
        }

        //printf("New Q-value: %lf\n", new_Qvalue);

        execute_back_propagation(neural_network, network_state, target_output);

        event = event->next_event;
        next_state_max_Qvalue = inv_sigmoid_function(get_max_value(target_output, network_state->output_size));
        //next_state_max_Qvalue = new_Qvalue;
        //next_state_max_Qvalue += reward;
    }
    free_event(history->event_head);
    history->event_head = NULL;
    history->size = 0;
}
