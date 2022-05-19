from typing import List
from ctypes import *


c_lib = cdll.LoadLibrary("./c_DQN/c_neural_network.so")


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


class c_Event(Structure):
    _fields_=[("network_state", POINTER(c_NetworkState)),
              ("chosen_action", c_int),
              ("reward", c_double),
              ("next_event", POINTER("c_Event"))]


class c_History(Structure):
    _fields_=[("size", c_int),
              ("event_head", c_Event)]


class NetworkState:
    def __init__(self, c_network_state) -> None:
        self.c_network_state = POINTER(c_NetworkState)
        self.c_network_state: c_NetworkState = c_network_state

    def __del__(self):
        if self.c_network_state:
            c_lib.free_network_state.argtype = POINTER(c_NetworkState)
            c_lib.free_network_state(self.c_network_state)

    def get_input_size(self) -> int:
        c_lib.get_input_size.argtype = POINTER(c_NetworkState)
        c_lib.get_input_size.restype = c_int
        return c_lib.get_input_size(self.c_network_state)

    def get_hidden_amount(self) -> int:
        c_lib.get_hidden_amount.argtype = POINTER(c_NetworkState)
        c_lib.get_hidden_amount.restype = c_int
        return c_lib.get_hidden_amount(self.c_network_state)

    def get_hidden_size(self) -> int:
        c_lib.get_hidden_size.argtype = POINTER(c_NetworkState)
        c_lib.get_hidden_size.restype = c_int
        return c_lib.get_hidden_size(self.c_network_state)

    def get_output_size(self) -> int:
        c_lib.get_output_size.argtype = POINTER(c_NetworkState)
        c_lib.get_output_size.restype = c_int
        return c_lib.get_output_size(self.c_network_state)

    def get_input_layer(self) -> List[float]:
        c_lib.get_input_layer.argtype = POINTER(c_NetworkState)
        c_lib.get_input_layer.restype = POINTER(c_double * self.input_size)
        input_layer = c_lib.get_input_layer(self.c_network_state)
        return [i for i in input_layer.contents]

    def get_hidden_layers(self) -> List[List[float]]:
        c_lib.get_hidden_layers.argtype = POINTER(c_NetworkState)
        c_lib.get_hidden_layers.restype = POINTER(POINTER(c_double * self.hidden_size) * self.hidden_amount)
        hidden_layers = c_lib.get_hidden_layers(self.c_network_state)
        return [i for i in hidden_layers.contents]

    def get_output_layer(self) -> List[float]:
        c_lib.get_output_layer.argtype = POINTER(c_NetworkState)
        c_lib.get_output_layer.restype = POINTER(c_double * self.get_output_size())
        output_layer = c_lib.get_output_layer(self.c_network_state)
        return [i for i in output_layer.contents]

    def get_output(self) -> List[float]:
        c_lib.get_output.argtype = POINTER(c_NetworkState)
        c_lib.get_output.restype = POINTER(c_double * self.get_output_size())
        output = c_lib.get_output(self.c_network_state)
        return [i for i in output.contents]

    def choose_action(self) -> int:
        c_lib.get_output.argtype = POINTER(c_NetworkState)
        c_lib.get_output.restype = c_int
        return c_lib.choose_action(self.c_network_state)

    def display_output(self) -> int:
        c_lib.get_output.argtype = POINTER(c_NetworkState)
        c_lib.display_output(self.c_network_state)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_amount: int, hidden_size: int, output_size: int, learning_rate: float, momentum_value: float, momentum_enabled: bool) -> None:
        self.c_neural_network = POINTER(c_NeuralNetwork)
        c_lib.create_network.argtypes = (c_int, c_int, c_int, c_int, c_double, c_double, c_int)
        c_lib.create_network.restype = POINTER(c_NeuralNetwork)
        c_momentum_enabled = 1 if momentum_enabled else 0
        self.c_neural_network = c_lib.create_network(input_size, hidden_amount, hidden_size, output_size, learning_rate, momentum_value, c_momentum_enabled)

    def __del__(self):
        c_lib.free_neural_network.argtype = POINTER(c_NeuralNetwork)
        c_lib.free_neural_network(self.c_neural_network)

    def execute_forward_propagation(self, input: List[float]) -> NetworkState:
        #return NetworkState(neural_network_c.execute_forward_propagation(self.network_c, input))
        len_input = len(input)
        c_lib.execute_forward_propagation.argtype = POINTER(c_double * len_input)
        c_lib.execute_forward_propagation.restype = POINTER(c_NetworkState)
        return NetworkState(c_lib.execute_forward_propagation(self.c_neural_network, (c_double * len_input)(*input)))

    def execute_back_progagation(self, network_state: NetworkState, target_output: List[float]) -> None:
        len_target = len(target_output)
        c_lib.execute_back_propagation.argtypes = (POINTER(c_NeuralNetwork), POINTER(c_NetworkState), POINTER(c_double * len_target))
        c_lib.execute_back_propagation(self.c_neural_network, network_state.c_network_state, (c_double * len_target)(*target_output))


class History:
    def __init__(self) -> None:
        self.c_history = POINTER(c_History)
        c_lib.create_history.restype = POINTER(c_History)
        self.c_history = c_lib.create_history()

    def __del__(self):
        c_lib.free_history.argtype = POINTER(c_History)
        c_lib.free_history(self.c_history)

    def get_length(self) -> int:
        c_lib.get_history_size.argtype = POINTER(c_History)
        c_lib.get_history_size.restype = c_int
        return c_lib.get_history_size(self.c_history)

    def add_event(self, network_state: NetworkState, chosen_action: int, reward: float) -> None:
        c_lib.add_event.argtypes = (POINTER(c_History), POINTER(c_NetworkState), c_int, c_double)
        c_lib.add_event(self.c_history, network_state.c_network_state, chosen_action, reward)
        network_state.c_network_state = None

    #def update_neural_network_pop_amount(self, neural_network: NeuralNetwork, pop_amount: int, alpha: float, gamma: float) -> None:
    #    c_lib.perform_batch_update_pop_amount.argtypes = (POINTER(c_NeuralNetwork), POINTER(c_History), c_int, c_double, c_double)
    #    c_lib.perform_batch_update_pop_amount(neural_network.c_neural_network, self.c_history, pop_amount, alpha, gamma)

    def update_neural_network_last_event(self, neural_network: NeuralNetwork, alpha: float, gamma: float) -> None:
        c_lib.perform_batch_update_last.argtypes = (POINTER(c_NeuralNetwork), POINTER(c_History), c_double, c_double)
        c_lib.perform_batch_update_last(neural_network.c_neural_network, self.c_history, alpha, gamma)

    def update_neural_network_all_events(self, neural_network: NeuralNetwork, alpha: float, gamma: float) -> None:
        c_lib.perform_batch_update_all.argtypes = (POINTER(c_NeuralNetwork), POINTER(c_History), c_double, c_double)
        c_lib.perform_batch_update_all(neural_network.c_neural_network, self.c_history, alpha, gamma)

