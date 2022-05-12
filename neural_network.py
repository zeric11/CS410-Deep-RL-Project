from typing import List
from ctypes import *


c_neural_network = cdll.LoadLibrary("./c_neural_network.so")


class c_NetworkState(Structure):
    _fields_=[("input_size", c_int),
              ("hidden_amount", c_int),
              ("hidden_size", c_int),
              ("output_size", c_int),
              ("input_layer", POINTER(c_double)),
              ("hidden_layers", POINTER(POINTER(c_double))),
              ("output_layer", POINTER(c_double)),
              ("input_weights", POINTER(POINTER(c_double))),
              ("hidden_weights", POINTER(POINTER(POINTER(c_double)))),
              ("output_weights", POINTER(POINTER(c_double)))]


class c_NeuralNetwork(Structure):
    _fields_=[("input_size", c_int),
              ("hidden_amount", c_int),
              ("hidden_size", c_int),
              ("output_size", c_int),
              ("input_weights", POINTER(POINTER(c_double))),
              ("hidden_weights", POINTER(POINTER(POINTER(c_double)))),
              ("output_weights", POINTER(POINTER(c_double))),
              ("delta_input_weights", POINTER(POINTER(c_double))),
              ("delta_hidden_weights", POINTER(POINTER(POINTER(c_double)))),
              ("delta_output_weights", POINTER(POINTER(c_double)))]


class NetworkState:
    def __init__(self, c_state) -> None:
        self.c_state = POINTER(c_NetworkState)
        self.c_state: c_NetworkState = c_state

        c_neural_network.get_input_size.argtype = POINTER(c_NetworkState)
        c_neural_network.get_input_size.restype = c_int
        self.input_size: int = c_neural_network.get_input_size(self.c_state)

        c_neural_network.get_hidden_amount.argtype = POINTER(c_NetworkState)
        c_neural_network.get_hidden_amount.restype = c_int
        self.hidden_amount: int = c_neural_network.get_hidden_amount(self.c_state)

        c_neural_network.get_hidden_size.argtype = POINTER(c_NetworkState)
        c_neural_network.get_hidden_size.restype = c_int
        self.hidden_size: int = c_neural_network.get_hidden_size(self.c_state)

        c_neural_network.get_output_size.argtype = POINTER(c_NetworkState)
        c_neural_network.get_output_size.restype = c_int
        self.output_size: int = c_neural_network.get_output_size(self.c_state)

        c_neural_network.get_input_layer.argtype = POINTER(c_NetworkState)
        c_neural_network.get_input_layer.restype = POINTER(c_double * self.input_size)
        input_layer = c_neural_network.get_input_layer(self.c_state)
        self.input_layer = [i for i in input_layer.contents]

        c_neural_network.get_hidden_layers.argtype = POINTER(c_NetworkState)
        c_neural_network.get_hidden_layers.restype = POINTER(POINTER(c_double * self.hidden_size) * self.hidden_amount)
        hidden_layers = c_neural_network.get_hidden_layers(self.c_state)
        self.hidden_layers = [i for i in hidden_layers.contents]

        c_neural_network.get_output_layer.argtype = POINTER(c_NetworkState)
        c_neural_network.get_output_layer.restype = POINTER(c_double * self.output_size)
        output_layer = c_neural_network.get_output_layer(self.c_state)
        self.output_layer = [i for i in output_layer.contents]

    def __del__(self):
        c_neural_network.free_network_state.argtype = POINTER(c_NetworkState)
        c_neural_network.free_network_state(self.c_state)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_amount: int, hidden_size: int, output_size: int) -> None:
        self.c_network = POINTER(c_NeuralNetwork)
        c_neural_network.create_network.argtypes = (c_int, c_int, c_int, c_int)
        c_neural_network.create_network.restype = POINTER(c_NeuralNetwork)
        self.c_network = c_neural_network.create_network(input_size, hidden_amount, hidden_size, output_size)

    def __del__(self):
        c_neural_network.free_neural_network.argtype = POINTER(c_NeuralNetwork)
        c_neural_network.free_neural_network(self.c_network)

    def execute_forward_propagation(self, input: List[float]) -> NetworkState:
        #return NetworkState(neural_network_c.execute_forward_propagation(self.network_c, input))
        len_input = len(input)
        c_neural_network.execute_forward_propagation.argtype = POINTER(c_double * len_input)
        c_neural_network.execute_forward_propagation.restype = POINTER(c_NetworkState)
        return NetworkState(c_neural_network.execute_forward_propagation(self.c_network, (c_double * len_input)(*input)))

    def execute_back_progagation(self, network_state: NetworkState, target_output: List[float]) -> None:
        len_target = len(target_output)
        c_neural_network.execute_back_propagation.argtype = (POINTER(c_NetworkState), POINTER(c_double * len_target))
        c_neural_network.execute_back_propagation(self.c_network, network_state.c_state, (c_double * len_target)(*target_output))



