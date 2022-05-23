#include "c_DQN.h"


int main() {
    printf("Running Test...");
    fflush(stdout);

    struct NeuralNetwork* neural_network = create_network(10, 5, 100, 10, 0.1, 0.5, 1);
    struct History* history = create_history();

    for(int i = 0; i < 10; ++i) {
        double input[10];
        for(register int j = 0; j < 10; ++j) {
            input[j] = 0;
        }

        struct NetworkState* network_state = execute_forward_propagation(neural_network, input);
        int chosen_action = choose_action(network_state);
        double reward = -10;

        int target_action = 0;
        if(i + 1 <= 100) {
            input[0] = 10;
        }
        if(i + 1 > 100) {
            input[1] = 10;
            target_action = 1;
        }
        if(i + 1 > 200) {
            input[2] = 10;
            target_action = 2;
        }
        if(i + 1 > 300) {
            input[3] = 10;
            target_action = 3;
        }
        if(i + 1 > 400) {
            input[4] = 10;
            target_action = 4;
        }
        if(i + 1 > 500) {
            input[5] = 10;
            target_action = 5;
        }
        if(i + 1 > 600) {
            input[6] = 10;
            target_action = 6;
        }
        if(i + 1 > 700) {
            input[7] = 10;
            target_action = 7;
        }
        if(i + 1 > 800) {
            input[8] = 10;
            target_action = 8;
        }
        if(i + 1 > 900) {
            input[9] = 10;
            target_action = 9;
        }
        if(i + 1 > 1000) {
            input[0] = 10;
            target_action = 0;
        }
        if(i + 1 > 1100) {
            input[1] = 10;
            target_action = 1;
        }
        if(i + 1 > 1200) {
            input[2] = 10;
            target_action = 2;
        }
        if(i + 1 > 1300) {
            input[3] = 10;
            target_action = 3;
        }
        if(i + 1 > 1400) {
            input[4] = 10;
            target_action = 4;
        }
        if(i + 1 > 1500) {
            input[5] = 10;
            target_action = 5;
        }
        if(i + 1 > 1600) {
            input[6] = 10;
            target_action = 6;
        }
        if(i + 1 > 1700) {
            input[7] = 10;
            target_action = 7;
        }
        if(i + 1 > 1800) {
            input[8] = 10;
            target_action = 8;
        }
        if(i + 1 > 1900) {
            input[9] = 10;
            target_action = 9;
        }

        if(chosen_action == target_action) {
            reward = 10;
        }

        printf("Step: %d, [", i + 1);
        display_output(network_state);
        printf("], action: %d, reward: %lf\n", chosen_action, reward);

        add_event(history, network_state, chosen_action, reward);
        
        if(get_history_size(history) >= 100) {
            perform_batch_update_last(neural_network, history, 0.5, 0.9);
        }
    }

    while(history->size > 0) {
        perform_batch_update_all(neural_network, history, 0.5, 0.9);
    }

    free_history(history);
    free_neural_network(neural_network);

    printf("Done.\n");

    return 0;
}