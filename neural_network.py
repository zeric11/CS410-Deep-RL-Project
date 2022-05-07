from typing import List
from ctypes import *


neural_network_c = cdll.LoadLibrary("./neural_network_c.so")


class NetworkState:
    def __init__(self, state_c) -> None:
        self.state_c = state_c

        self.input_size: int = neural_network_c.get_input_size(self.state_c)
        self.hidden_amount: int = neural_network_c.get_hidden_amount(self.state_c)
        self.hidden_size: int = neural_network_c.get_hidden_size(self.state_c)
        self.output_size: int = neural_network_c.get_output_size(self.state_c)

        neural_network_c.get_input_layer.restype = POINTER(c_double * self.input_size)
        neural_network_c.get_hidden_layers.restype = POINTER(POINTER(c_double * self.hidden_size) * self.hidden_amount)
        neural_network_c.get_output_layer.restype = POINTER(c_double * self.output_size)

        input_layer = neural_network_c.get_input_layer(self.state_c)
        hidden_layers = neural_network_c.get_hidden_layers(self.state_c)
        output_layer = neural_network_c.get_output_layer(self.state_c)

        self.input_layer = [i for i in input_layer.contents]
        self.hidden_layers = [i for i in hidden_layers.contents]
        self.output_layer = [i for i in output_layer.contents]

    def __del__(self):
        neural_network_c.free_network_state(self.state_c)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_amount: int, hidden_size: int, output_size: int) -> None:
        self.network_c = neural_network_c.create_network(input_size, hidden_amount, hidden_size, output_size)

    def __del__(self):
        neural_network_c.free_neural_network(self.network_c)

    def execute_forward_propagation(self, input: List[float]) -> NetworkState:
        #return NetworkState(neural_network_c.execute_forward_propagation(self.neural_network, input))
        return NetworkState(neural_network_c.execute_forward_propagation(self.network_c, (c_double * len(input))(*input)))

    def execute_back_progagation(self, network_state: NetworkState, target_output: List[float]) -> None:
        neural_network_c.execute_back_propagation(self.network_c, network_state.state_c, (c_double * len(target_output))(*target_output))



