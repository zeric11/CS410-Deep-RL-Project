# Compile for Python
dqn: c_neural_network.c c_history.c utils.c
	gcc -O0 -pthread -fPIC -shared c_neural_network.c c_history.c utils.c -o c_dqn.so -lpthread -lm


# Compile for Testing
# Run with: valgrind --tool=memcheck --leak-check=yes -s ./dqn_test
dqn_test: testing.c c_neural_network.c c_history.c utils.c
	gcc -O0 -pthread -g testing.c c_neural_network.c c_history.c utils.c -o c_dqn_test -lpthread -lm


clean_dqn:
	rm c_dqn.so


clean_dqn_test:
	rm c_dqn_test
