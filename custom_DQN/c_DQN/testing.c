#include "c_DQN.h"


int main() {
    printf("Running Test...");
    fflush(stdout);

    struct NeuralNetwork* neural_network = create_network(210 * 160 * 2, 1, 50, 4, 0.1, 0.9, 0);
    struct History* history = create_history();

    for(int i = 0; i < 100; ++i) {
        double input[210 * 160 * 2];
        for(register int j = 0; j < 210 * 160 * 2; ++j) {
            input[j] = 255;
        }

        struct NetworkState* network_state = execute_forward_propagation(neural_network, input);
        double* output = get_output(network_state);
        int chosen_action = get_max_index(output, get_output_size(network_state));
        double reward = -10;

        if(chosen_action == 1) {
            reward = 10;
        }

        printf("Step: %d, [", i + 1);
        display_output(network_state);
        printf("], action: %d, reward: %lf\n", chosen_action, reward);

        add_event(history, network_state, chosen_action, reward);
        
        if(get_history_size(history) >= 30) {
            perform_batch_update_last(neural_network, history, 0.1, 0.9);
        }
    }

    while(history->size > 0) {
        perform_batch_update_all(neural_network, history, 0.1, 0.9);
    }

    free_history(history);
    free_neural_network(neural_network);

    printf("Done.\n");

    return 0;
}