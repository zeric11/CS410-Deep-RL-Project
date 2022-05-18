CC=gcc
CFLAGS=-O3.


# Compile for Python
dqn: c_neural_network.c c_history.c utils.c
	$(CC) -fPIC -shared c_neural_network.c c_history.c utils.c -o c_neural_network.so -lm


# Compile for Testing
# Run with: valgrind --tool=memcheck --leak-check=yes -s ./c_neural_network
dqn_testing: testing.c c_neural_network.c c_history.c utils.c
	$(CC) -g testing.c c_neural_network.c c_history.c utils.c -o c_neural_network -lm


clean:
	rm c_neural_network.so c_neural_network
