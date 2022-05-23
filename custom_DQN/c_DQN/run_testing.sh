make dqn_test
valgrind --tool=memcheck --leak-check=yes -s ./c_dqn_test
make clean_dqn_test
