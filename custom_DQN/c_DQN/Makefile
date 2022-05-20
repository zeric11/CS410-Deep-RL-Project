# Compile for Python
dqn: c_neural_network.c c_history.c utils.c
	gcc -O3 -pthread -fPIC -shared c_neural_network.c c_history.c utils.c -o c_neural_network.so -lpthread -lm


# Compile for Testing
# Run with: valgrind --tool=memcheck --leak-check=yes -s ./dqn_test
dqn_testing: testing.c c_neural_network.c c_history.c utils.c
	gcc -O3 -pthread -g testing.c c_neural_network.c c_history.c utils.c -o dqn_test -lpthread -lm


clean:
	rm c_neural_network.so c_neural_network
